from openai import OpenAI
import os
import logging
# from duckduckgo_search import DDGS # Commented out DDGS
from googlesearch import search as google_search_func
from memory import Memory # Added Memory import
from pdf_vision_assistant import PDFVisionAssistant # For shared memory demo
import re
import time
from urllib.parse import urlparse
import concurrent.futures
import markdown
from weasyprint import HTML as WeasyHTML

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent"""
        # Use environment variable or configuration file for API key in production
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # Fetch API key from environment variable
            base_url=os.getenv("DEEPSEEK_BASE_URL")  # Fetch base URL from environment variable
        )
        self.memory = Memory() # Initialize Memory
        self.report_sections = []
        # search_history and reference_map will be loaded from memory per topic in research_cycle
        self.search_history = []
        self.reference_index = 1
        self.reference_map = {}
        self.initial_topic = ""
        self.research_mode = "general" # Default research mode
        
    def web_search(self, query: str, num_results: int = 15, inter_search_delay: int = 2) -> list:
        """Perform comprehensive web search using Google Search with delays between search types."""
        logger.info(f"🔍 Google Search: {query} (aiming for ~{num_results} base results per category)")
        results = []
        all_g_results_combined = []
        seen_urls = set()

        # 1. General Google Search
        try:
            logger.info(f"   -> General search for: {query}")
            general_g_results = list(google_search_func(query, num_results=num_results, lang='en', advanced=True, sleep_interval=1))
            all_g_results_combined.extend(general_g_results)
        except Exception as e:
            logger.error(f"Error during General Google Search for '{query}': {str(e)}")

        time.sleep(inter_search_delay) # Delay before next search type

        # 2. Google News-like Search (append "news" to query)
        try:
            news_query = f"{query} news"
            logger.info(f"   -> News-like search for: {news_query}")
            news_g_results = list(google_search_func(news_query, num_results=max(1, num_results // 2 + 3), lang='en', advanced=True, sleep_interval=1))
            all_g_results_combined.extend(news_g_results)
        except Exception as e:
            logger.error(f"Error during News-like Google Search for '{news_query}': {str(e)}")

        time.sleep(inter_search_delay) # Delay before next search type

        # 3. Google Academic-like Search (using site operators)
        try:
            academic_query = f"site:.gov OR site:.edu OR site:arxiv.org OR site:ieee.org OR site:acm.org OR site:scholar.google.com {query}"
            logger.info(f"   -> Academic-like search for: {academic_query}")
            academic_g_results = list(google_search_func(academic_query, num_results=max(1, num_results // 2 + 3), lang='en', advanced=True, sleep_interval=1))
            all_g_results_combined.extend(academic_g_results)
        except Exception as e:
            logger.error(f"Error during Academic-like Google Search for '{academic_query}': {str(e)}")
        
        if not all_g_results_combined:
            logger.warning(f"Google Search yielded no results for query: '{query}'.")
            return []

        logger.info(f"Processing {len(all_g_results_combined)} initial items from Google.")
        for g_res in all_g_results_combined:
            try:
                if not hasattr(g_res, 'url') or not g_res.url:
                    continue 
                url = g_res.url
                if url in seen_urls:
                    continue
                
                title = g_res.title if hasattr(g_res, 'title') else 'No title available'
                snippet = g_res.description if hasattr(g_res, 'description') else 'No snippet available'
                if not snippet and hasattr(g_res, 'text'): # Fallback for snippet if description is empty
                    snippet = g_res.text
                if not title and snippet: # Fallback for title from snippet
                     title = snippet.split('.')[0] if '.' in snippet else snippet[:50]

                domain = self._extract_domain(url)
                credibility = self._assess_credibility(domain, url)
                
                is_academic = any(d in url.lower() for d in ['.edu', '.ac.', '.gov', 'arxiv', 'ieee', 'acm.org', 'scholar.google'])
                # A bit more heuristic for news, checking domain and title if Google result was from news_query
                is_news_heuristic = any(d in domain.lower() for d in ['news', 'times', 'post', 'guardian', 'reuters', 'bbc', 'cnn', 'ap', 'wsj', 'ft'])
                if not is_news_heuristic and title and any(kw in title.lower() for kw in ['news', 'report', 'update', 'breaking']):
                    is_news_heuristic = True
                
                result_type = "academic" if is_academic else "news" if is_news_heuristic else "general"
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "domain": domain,
                    "credibility": credibility,
                    "type": result_type
                })
                seen_urls.add(url)
            except Exception as e:
                item_title = g_res.title if hasattr(g_res, 'title') else 'N/A'
                logger.warning(f"Warning: Failed to process Google result item '{item_title}': {str(e)}")
                continue

        if not results:
            logger.warning(f"Processing Google Search for '{query}' yielded no usable results.")
            return []

        results.sort(key=lambda x: x['credibility']['total'], reverse=True)
        # Take up to num_results * 2 because we combined three searches with increased individual aims.
        final_results = results[:int(num_results * 2)]
        # Update search history in memory
        self.search_history.append({"query": query, "results": [r['url'] for r in final_results]})
        if self.initial_topic: # Ensure initial_topic is set before saving
            self.memory.add_to_memory("search_histories", self.initial_topic, self.search_history)
        logger.info(f"Selected {len(final_results)} results for query '{query}' after processing and sorting.")
        return final_results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name (e.g., 'google' from 'www.google.com')"""
        try:
            parsed_url = urlparse(url)
            netloc = parsed_url.netloc
            if not netloc:
                # Fallback for URLs like 'file:///path/to/file' or malformed URLs
                path_parts = parsed_url.path.split('/')
                if len(path_parts) > 1 and '.' in path_parts[-1]: # Check if last part looks like a file with extension
                    return path_parts[-2] if len(path_parts) > 2 else 'localfile'
                return 'unknown_domain'
            # Remove 'www.' and select the part before the first dot of the main domain part
            domain_parts = netloc.replace("www.", "").split('.')
            return domain_parts[0] if domain_parts else netloc # Handle cases like 'localhost'
        except Exception as e:
            logger.error(f"Error extracting domain from URL '{url}': {e}")
            return "unknown_domain"

    def analyze_results(self, query: str, results: list) -> str:
        """Analyze search results to extract key information and identify themes for a research section."""
        if not results:
            logger.warning(f"No results to analyze for query: {query}. Generating placeholder analysis.")
            return f"""# Analysis for: {query}

No search results were available to analyze for this specific query. This could be due to the specificity of the query, limitations in search depth, or lack of available online information matching the criteria.

Potential next steps:
- Broaden the search terms.
- Explore related concepts or tangential queries.
- Consult offline resources or specialized databases if the topic is highly niche.

Follow-up Research Questions:
1. What alternative search strategies could yield results for '{query}'?
2. Are there analogous topics that have been researched and could offer insights?
"""
        
        context_items = []
        for res_type in ["academic", "technical", "news", "general"]:
            typed_results = [r for r in results if r.get('type') == res_type and r.get('snippet')]
            if typed_results:
                context_items.append(f"\n## {res_type.capitalize()} Sources:")
                for res in typed_results[:5]: # Limit to top 5 of each type for concise analysis prompt
                    context_items.append(
                        f"- Title: {res['title']}\n"
                        f"  URL: {res['url']}\n"
                        f"  Snippet: {res['snippet']}\n"
                        f"  Credibility: {res['credibility']['total']}/100 ({res['credibility']['domain_type']})\n"
                    )
        
        if not context_items:
            return f"# Analysis for: {query}\n\nWhile search results were found, they could not be categorized or did not contain sufficient snippets for analysis."

        context = "\n".join(context_items)
        
        try:
            prompt = ""
            if self.research_mode == "market_research":
                company_name_from_query = self.initial_topic.split(" and its competitors")[0].strip()
                prompt = f"""Please perform a market research analysis of the following search results related to the query: '{query}'. The overall research is focused on {company_name_from_query} and its competitive landscape.

SOURCE MATERIAL:
{context}

MARKET ANALYSIS TASK:
Synthesize the information to build a market understanding. Focus on:

1.  **Company Information (if query is company-specific for {company_name_from_query} or a competitor):** Products/services, target market, key strengths/weaknesses, recent news/developments.
2.  **Competitor Identification & Analysis (if query is about competitors):** Identify key competitors of {company_name_from_query}. For each, briefly note their offerings, market position, and perceived strengths/weaknesses relative to {company_name_from_query}.
3.  **Market Overview/Trends (if query is about the market):** Describe the current state of the industry relevant to {company_name_from_query}. Identify key market trends, drivers, challenges, and opportunities.
4.  **SWOT Elements (if query is for SWOT or relevant info appears):** Identify Strengths, Weaknesses, Opportunities, and Threats relevant to {company_name_from_query} or its market based on the provided sources.
5.  **Customer Sentiments/Reviews (if available):** Note any mentions of customer feedback or reviews regarding {company_name_from_query} or its competitors.

OUTPUT REQUIREMENTS:
-   Provide a structured analysis using markdown.
-   Focus on actionable insights relevant to market research.
-   If the query was '{query}', clearly address how the sources inform that specific aspect of the market research for {company_name_from_query}.
-   Conclude with 2-3 "Key Market Insights or Follow-up Questions" for further investigation related to '{query}' and {company_name_from_query}.

Begin your market analysis now:
# Market Analysis of: {query}
"""
            else: # General research mode
                prompt = f"""Please perform a detailed analysis of the following search results related to the query: '{query}'.

SOURCE MATERIAL:
{context}

ANALYSIS TASK:
Your goal is to synthesize the information from these sources to build a comprehensive understanding of the topic. Structure your analysis to address the following, as applicable:

1.  **Key Themes and Concepts:** Identify the central ideas, theories, or findings presented across multiple sources. What are the recurring patterns or dominant narratives?
2.  **Evidence and Support:** What kind of evidence (e.g., empirical data, case studies, expert opinions, theoretical arguments) is provided for the key themes? Evaluate the strength and consistency of this evidence.
3.  **Different Perspectives/Contradictions:** Are there conflicting viewpoints, debates, or contradictory findings among the sources? If so, describe them.
4.  **Methodologies (if apparent):** If the sources describe research, what methodologies were used? Are there any notable strengths or weaknesses?
5.  **Gaps or Unanswered Questions:** Based on these sources, what aspects of the topic remain unclear or require further investigation? What questions arise from this initial review?
6.  **Connections and Relationships:** How do the different pieces of information relate to each other? Are there causal links, correlations, or hierarchical relationships?
7.  **Significance and Implications:** What is the broader significance of these findings or discussions? What are the potential implications (e.g., for policy, practice, future research)?

OUTPUT REQUIREMENTS:
-   Provide a structured analysis, using markdown for clarity.
-   Synthesize information rather than just summarizing individual sources.
-   Maintain a neutral, objective tone.
-   Clearly cite information by referring to source titles or URLs if specific details are drawn.
-   Conclude with a list of 3-5 specific and actionable "Follow-up Research Questions" that emerge from your analysis and would help to deepen the understanding of '{query}'.

Example of a Follow-up Question:
   - How does [specific finding from source X] correlate with [concept from source Y] in the context of [overall topic]?

Begin your analysis now:
# Detailed Analysis of: {query}
"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3500, 
                temperature=0.3 
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during AI analysis for '{query}': {str(e)}")
            return f"# Analysis Error for: {query}\n\nAn error occurred during the AI-powered analysis. Raw context provided to AI:\n{context}\n\nThis may be due to API issues or prompt complexity. Consider simplifying the query or checking API status."

    def _register_references(self, content: str) -> str:
        """Replace URLs with reference numbers and build citation map. Handles markdown links."""
        if not content:
            return ""
            
        try:
            # Regex to find markdown links [text](url) and plain URLs
            # It captures the full markdown link (group 1), the link text (group 2), and the URL (group 3)
            # Or it captures a plain URL (group 4)
            url_pattern = r'(\[([^\]]+)\]\((https?://[^\s\)\]]+)\))|(https?://[^\s\)\]]+)'
            
            matches = list(re.finditer(url_pattern, content))
            modified_content = content
            offset = 0 # To adjust indices after replacements

            for match in matches:
                full_match_str = match.group(0)
                markdown_link_full = match.group(1) # Entire [text](url)
                link_text = match.group(2)          # Just the text part
                url_in_markdown = match.group(3)    # URL from markdown link
                plain_url = match.group(4)          # Plain URL

                actual_url = url_in_markdown or plain_url
                if not actual_url:
                    continue

                clean_url = actual_url.rstrip('.,;:!?')
                
                existing_ref_num = None
                for ref_num, ref_details in self.reference_map.items():
                    if ref_details['url'] == clean_url:
                        existing_ref_num = ref_num
                        break
                
                if existing_ref_num is None:
                    current_ref_num = self.reference_index
                    # Try to get a title from link_text or from the URL itself if it's a plain URL
                    title_for_ref = link_text if link_text else f"Source document from {self._extract_domain(clean_url)}"
                    self.reference_map[current_ref_num] = {'url': clean_url, 'title': title_for_ref}
                    citation_text = f"[{current_ref_num}]"
                    self.reference_index += 1
                    # Save updated reference_map to memory
                    if self.initial_topic:
                        self.memory.add_to_memory("reference_maps", self.initial_topic, self.reference_map)
                else:
                    citation_text = f"[{existing_ref_num}]"
                
                # Replace the original URL (or markdown link) with the citation
                # Adjust match start and end due to previous replacements
                match_start = match.start() + offset
                match_end = match.end() + offset
                
                # If it was a markdown link, we want to keep the link text and replace the URL part effectively
                # For a plain URL, we replace the whole URL string.
                if markdown_link_full: # If it's a markdown link like [text](url)
                    # We want to transform [text](url) to text [ref]
                    # However, simple replacement might be tricky. Let's try keeping text and adding ref.
                    # For simplicity, we'll replace the whole markdown link with "link_text [ref]"
                    replacement_str = f"{link_text} {citation_text}"
                else: # If it's a plain URL
                    replacement_str = citation_text

                modified_content = modified_content[:match_start] + replacement_str + modified_content[match_end:]
                offset += len(replacement_str) - len(full_match_str)
            
            return modified_content
        except Exception as e:
            logger.error(f"Error in reference processing for content snippet: '{content[:100]}...': {str(e)}")
            return content # Return original content on error

    def generate_report_section_content(self, section_title: str, analysis_content: str, section_type: str) -> str:
        """Generates specific content for a report section based on its type (e.g., Introduction, Methodology)."""
        if not analysis_content and section_type not in ["Abstract", "Methodology", "Conclusion", "References", "Market_Overview", "Company_Profile", "Competitive_Landscape", "SWOT_Analysis", "Market_Outlook"]:
            return f"## {section_title}\n\n_Content for this section could not be generated due to missing prior analysis for relevant queries._"

        prompts = {}
        if self.research_mode == "market_research":
            company_name = self.initial_topic.split(" and its competitors")[0].strip()
            # Market research reports are typically more concise. Adjusted word counts for focused content.
            prompts = {
                "Market_Overview": f'''Write a Market Overview section for {company_name}\'s industry, based on the analysis for \'{section_title}\'.
Analysis:
{analysis_content}
Focus on: Market size, growth rate, key segments, industry trends, and overall attractiveness. Aim for approximately 400-500 words.''',
                "Company_Profile": f'''Write a detailed Company Profile for {company_name}, based on the analysis for \'{section_title}\'.
Analysis:
{analysis_content}
Focus on: Company history, mission, products/services, target audience, financial performance (if available), key executives, and recent significant developments. Aim for approximately 400-500 words.''',
                "Competitive_Landscape": f'''Describe the Competitive Landscape for {company_name}, based on the analysis for \'{section_title}\'.
Analysis:
{analysis_content}
Identify key competitors. For each, discuss their market share (if known), strengths, weaknesses, strategies, and how they compare to {company_name}. Aim for approximately 500-600 words.''',
                "SWOT_Analysis": f'''Conduct a SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats) for {company_name}, based on the analysis for \'{section_title}\'.
Analysis:
{analysis_content}
Clearly list and elaborate on each of the four components. Ensure it\'s well-supported by the analysis. Aim for approximately 300-400 words.''',
                "Market_Outlook": f'''Provide a Market Outlook for {company_name}\'s industry.
Synthesize all prior analyses including market overview, competitive landscape, and {company_name}\'s profile/SWOT.
Discuss: Future trends, potential disruptions, growth opportunities for {company_name}, and strategic recommendations. Aim for approximately 400-500 words.''',
                "Methodology": f'''Write the Methodology section for a market research report on \'{self.initial_topic}\'.
This report was generated by an AI research agent by:
1. Defining target company: {company_name}.
2. Performing targeted Google searches for: Company Profile, Market Overview, Competitors, SWOT analysis elements.
3. Analyzing search results using an AI model to extract relevant market intelligence.
4. Synthesizing this intelligence into sections: Market Overview, Company Profile for {company_name}, Competitive Landscape, SWOT Analysis, and Market Outlook.
Describe this process formally. Aim for approximately 200-250 words.''',
                "Conclusion": f'''Write a Conclusion for the market research report on {company_name}.
Summarize key findings from the Market Overview, Company Profile, Competitive Landscape, and SWOT Analysis. Reiterate the market outlook and key strategic recommendations for {company_name}. Aim for approximately 250-300 words.''',
            }
        else: # General Research Prompts (Targeting ~30 pages / 6000-7500 words total)
            prompts = {
                "Abstract": f'''Generate a comprehensive Abstract (around 200-250 words) for an in-depth research report on \'{self.initial_topic}\'.
Key themes identified include: {analysis_content if section_type == 'Abstract' else 'multiple complex facets of the topic'}.
The report aims to provide a thorough, multi-faceted overview, synthesize diverse findings, critically evaluate evidence, and identify significant areas for future scholarly investigation.
Based on the overall research topic \'{self.initial_topic}\', write the Abstract, emphasizing depth and breadth of coverage.''',
                "Introduction": f'''Write a detailed Introduction (aim for 700-1000 words) for an in-depth research report on \'{self.initial_topic}\'.
This section specifically addresses the sub-topic: \'{section_title}\'.
Raw analysis related to \'{section_title}\':
{analysis_content}
Content to include:
1.  **Comprehensive Background:** Thoroughly introduce \'{self.initial_topic}\', its historical context, and its contemporary significance. Establish a strong foundation for understanding.
2.  **Detailed Problem Statement/Rationale:** Articulate precisely why in-depth research on this topic is crucial. What specific gaps, controversies, or needs does this report aim to address?
3.  **Clear Research Questions/Objectives:** Clearly state the main research questions and subsidiary objectives that this section (derived from \'{section_title}\') and the overall report are trying to answer. Ensure these are analytical and not merely descriptive.
4.  **Scope and Delimitations:** Clearly define the boundaries of this section\'s analysis and the overall report. Specify what is included and excluded, and justify these choices.
5.  **Theoretical Framework (if applicable):** Briefly outline any theoretical lenses or frameworks guiding the research.
6.  **Report Roadmap:** Provide a clear overview of the report\'s structure and how each section contributes to the overall objectives.
Focus on expanding the raw analysis into a formal, extensive, and critical introduction for the section \'{section_title}\' within the broader report on \'{self.initial_topic}\'. Ensure a sophisticated academic tone and precise articulation.''',
                "Methodology": f'''Write a detailed Methodology section (aim for 500-700 words) for an in-depth research report on \'{self.initial_topic}\'.
Process: Iterative Google searches (general, news-specific, academic-specific queries), source credibility assessment (algorithmic scoring based on domain type and known reliable sources), AI-driven analysis of search results (identifying key themes, evidence, counter-arguments, methodological approaches, research gaps), and AI-powered synthesis into structured report sections.
Number of search cycles performed: {len(set(q['query'] for q in self.search_history)) if self.search_history else 'multiple'}.
Elaborate on:
- The rationale for using AI-assisted iterative searching.
- Details of the source selection and credibility filtering process.
- How the AI model is prompted to analyze and synthesize information for different sections (e.g., distinguishing between findings and discussion).
- The process of reference management.
- Limitations of this AI-driven research methodology (e.g., potential biases in search algorithms, AI interpretation limitations, reliance on digitally available data).''',
                "Results/Findings": f'''Based on the detailed analysis for query \'{section_title}\':
{analysis_content}
Synthesize the key findings into a comprehensive \'Results/Findings\' section. This section should be substantial, aiming for 1000-1200 words per major research query/theme.
REQUIREMENTS:
-   Present findings objectively, systematically, and with significant detail. Use subheadings extensively to organize different themes, sub-themes, and specific pieces of evidence.
-   Focus on *what* was found. Go beyond surface-level summaries; present data, examples, and supporting details extracted from the analysis.
-   Ensure a logical flow, connecting related findings smoothly.
-   Maintain a formal, academic tone with precise language.
-   If the analysis mentions specific sources or data points, ensure these are clearly presented (references will be handled by the citation system).
-   Avoid extensive interpretation or discussion here; that belongs in the \'Discussion\' section.''',
                "Discussion": f'''Write an extensive Discussion section (aim for 1000-1200 words) for the research findings related to \'{section_title}\'. The raw analysis that led to these findings is below:
{analysis_content}
CONTENT TO INCLUDE (elaborate significantly on each):
1.  **In-depth Interpretation of Findings:** What do the results (derived from the analysis above) signify in a broader context? Explore nuances and complexities.
2.  **Critical Comparison with Existing Literature/Theory:** How do these findings align, contrast, or extend existing scholarly work or theoretical frameworks related to \'{self.initial_topic}\' or specifically \'{section_title}\'? (If direct literature comparison is hard for AI, focus on theoretical implications or consistency with established concepts).
3.  **Strengths and Limitations of the Evidence/Analysis:** Critically evaluate the quality and nature of the information gathered for \'{section_title}\'. Discuss limitations of the AI\'s analysis process for this specific query.
4.  **Profound Implications:** What are the far-reaching practical, theoretical, societal, or policy implications of the findings? Explore multiple dimensions.
5.  **Substantial Suggestions for Future Research:** Based *specifically* on the gaps and insights from the findings for \'{section_title}\', propose several detailed and well-justified avenues for future scholarly investigation.
Ensure this section critically examines the findings with depth and intellectual rigor.''',
                "Conclusion": f'''Write a comprehensive Conclusion (aim for 500-700 words) for the in-depth research report on \'{self.initial_topic}\'.
Key analyses included: {', '.join([s['title'] for s in self.report_sections if s['title'] not in ['Abstract','Methodology','Conclusion']][:5])}...
CONTENT TO INCLUDE:
1.  **Synthesis of Core Findings:** Concisely synthesize the most critical findings from across all Results/Findings sections, drawing connections between them.
2.  **Revisiting Research Questions/Objectives:** Explicitly address how the report (and its constituent sections) answered the initial research questions and met the objectives.
3.  **Overall Significance and Contribution to Knowledge:** Clearly articulate the main takeaway and the report\'s overall contribution to understanding \'{self.initial_topic}\'.
4.  **Reflection on Methodological Limitations:** Briefly reflect on the limitations of the AI-driven research process as a whole and their potential impact on the outcomes.
5.  **Broad Recommendations & Forward-Looking Statement:** Provide overarching recommendations stemming from the entire body of research. Conclude with a forward-looking statement about the future of the topic or research in this area.''',
            }

        prompt_template = prompts.get(section_type)
        if not prompt_template:
            logger.warning(f"Warning: No specific prompt template for section type \'{section_type}\'. Using generic approach for \'{section_title}\'.")
            # Fallback for custom sections, assuming analysis_content is the main source
            prompt_template = f'''Write a detailed report section titled \'{section_title}\'.
Based on the following analysis or information:

{analysis_content}

Ensure the section is comprehensive, well-structured, and maintains an academic tone. Aim for approximately 400-500 words.'''

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt_template}],
                max_tokens=3000, # Maximize token allocation for detailed content
                temperature=0.4 # Slightly higher temp for more elaborate writing
            )
            generated_content = response.choices[0].message.content
            # Always register references for any generated content that might contain URLs from prompts
            return self._register_references(generated_content) 
        except Exception as e:
            logger.error(f"Error generating content for section '{section_title}' (type: {section_type}): {str(e)}")
            return f"## {section_title}\n\n_Content generation for this section failed due to an API error. Analysis provided to AI:_\n{analysis_content[:500]}..." 

    def _assess_credibility(self, domain: str, url: str) -> dict:
        """Score source credibility based on domain type and known sources. More nuanced scoring."""
        try:
            score = {'authority': 0, 'domain_type': 'unknown', 'total': 0, 'freshness_bonus': 0}
            domain_lower = domain.lower()
            url_lower = url.lower()

            # Domain Type Scoring (base authority)
            if any(d in url_lower for d in ['.gov', '.edu', '.ac.']): score.update({'authority': 90, 'domain_type': 'academic/governmental'})
            elif any(d in url_lower for d in ['ieee.org', 'acm.org', 'arxiv.org', 'nature.com', 'sciencemag.org', 'thelancet.com', 'nejm.org', 'cell.com']): score.update({'authority': 95, 'domain_type': 'top-tier academic/journal'})
            elif any(d in url_lower for d in ['.org', 'research.', 'foundation']): score.update({'authority': 70, 'domain_type': 'organization/research'})
            elif any(d in domain_lower for d in ['wikipedia']): score.update({'authority': 50, 'domain_type': 'encyclopedic', 'notes': 'Good starting point, but verify with primary sources.'})
            elif any(d in url_lower for d in ['.com', '.net', '.co', '.io', '.ai', '.news', '.info']): score.update({'authority': 50, 'domain_type': 'commercial/general/news'})
            else: score['authority'] = 30

            # Specific High-Credibility Source Bonus (stacks with domain type)
            high_cred_keywords = {'nature': 10, 'science': 10, 'ieee': 8, 'acm': 8, 'arxiv': 5, 'pubmed': 8, 'lancet': 10, 'nejm': 10, 'cell': 10, 'mit.edu': 5, 'stanford.edu': 5, 'harvard.edu': 5, 'ox.ac.uk': 5, 'cam.ac.uk': 5}
            for keyword, bonus in high_cred_keywords.items():
                if keyword in url_lower: score['authority'] = min(100, score['authority'] + bonus)
            
            # Penalty for certain indicators (blogs, forums, personal sites unless clearly academic)
            low_cred_indicators = ['blog', 'forum', 'discussion', 'personal', 'opinion', 'wordpress.com', 'blogspot.com']
            if score['domain_type'] not in ['top-tier academic/journal', 'academic/governmental'] and any(indicator in url_lower for indicator in low_cred_indicators):
                score['authority'] = max(10, score['authority'] - 20)
                if 'notes' not in score: score['notes'] = ""
                score['notes'] += " Potential lower credibility indicators found."

            # Freshness (very basic, could be improved with actual date parsing from content if available)
            # This is a placeholder as we don't parse dates yet.
            # if "news" in score['domain_type']: score['freshness_bonus'] = 5 

            score['total'] = min(100, score['authority'] + score['freshness_bonus'])
            return score
        except Exception as e:
            logger.error(f"Error in credibility assessment for {url}: {str(e)}")
            return {'authority': 0, 'domain_type': 'error', 'total': 0}

    def research_cycle(self, topic: str, depth: int = 3, research_mode: str = "general", per_query_delay: int = 5):
        """Execute research cycles, generate analysis, and then compile the report."""
        logger.info(f"🚀 Initializing Research For Topic: {topic} (Mode: {research_mode})")
        self.initial_topic = topic
        self.research_mode = research_mode # Set the mode for this cycle

        # Load existing data from memory for this topic
        logger.info(f"Attempting to load past data for topic '{topic}' from memory.")
        self.search_history = self.memory.get_from_memory("search_histories", self.initial_topic) or []
        self.reference_map = self.memory.get_from_memory("reference_maps", self.initial_topic) or {}

        if self.reference_map:
            # Ensure keys are integers for max() when loaded from JSON
            int_keys = [int(k) for k in self.reference_map.keys()]
            self.reference_index = max(int_keys, default=0) + 1
            logger.info(f"Loaded {len(self.reference_map)} references. Next reference index: {self.reference_index}")
        else:
            self.reference_index = 1
            logger.info("No existing references found for this topic. Starting fresh.")

        if self.search_history:
            logger.info(f"Loaded {len(self.search_history)} past search queries for this topic.")

        # Check if a report for this topic already exists
        existing_report_info = self.memory.get_from_memory("research_reports", topic)
        if existing_report_info:
            report_timestamp = existing_report_info.get("timestamp", "N/A")
            report_mode = existing_report_info.get("mode", "N/A")
            logger.info(f"Note: A report for '{topic}' (mode: {report_mode}) created on {report_timestamp} already exists in memory.")
            # Potentially add logic here to ask user if they want to overwrite or use existing. For now, just a note.

        self.report_sections = [] # Reset for new research generation cycle (content itself is from analysis of queries)

        queries = [topic] # Start with the main topic
        # For market research, we might want a more structured set of initial queries
        if self.research_mode == "market_research":
            company_name = topic.split(" and its competitors")[0].strip() # Basic extraction
            queries = [
                f"Company profile of {company_name}",
                f"Market overview for {company_name}'s industry",
                f"Competitors of {company_name}",
                f"SWOT analysis for {company_name}",
                # Add more specific market research queries if needed
            ]
            # We might only want to do 1 depth cycle for these structured queries
            depth = 1 
            logger.info(f"Market research mode: using structured queries: {queries}")

        explored_queries = set()
        max_queries_per_cycle = 2 if self.research_mode == "general" else len(queries) # Process all structured queries in market mode

        for cycle_num in range(depth):
            logger.info(f"\n📚 Research Cycle {cycle_num + 1}/{depth}")
            current_cycle_queries = queries[:max_queries_per_cycle]
            next_cycle_potential_queries = []
            
            if not current_cycle_queries:
                logger.info("No more queries to process in this cycle.")
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_queries_per_cycle) as executor:
                future_to_query = {executor.submit(self.process_single_query_for_analysis, query, cycle_num): query for query in current_cycle_queries if query not in explored_queries}
                
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        analysis_content, follow_up_qs = future.result()
                        if analysis_content:
                            self.report_sections.append({'title': query, 'raw_analysis': analysis_content, 'content': ''})
                            explored_queries.add(query)
                            if follow_up_qs:
                                next_cycle_potential_queries.extend(q for q in follow_up_qs if q not in explored_queries and q not in next_cycle_potential_queries)
                        else:
                             logger.warning(f"No analysis generated for query: {query}")
                    except Exception as e:
                        logger.error(f"Error processing future for query '{query}' in cycle {cycle_num + 1}: {str(e)}")
                    finally:
                        time.sleep(per_query_delay) # Delay after processing each query
            
            if not next_cycle_potential_queries and self.research_mode == "general":
                logger.info("✓ No new follow-up queries identified. Concluding research cycles.")
                break
            
            # Simple query selection: take unique new queries, prioritizing those not yet explored.
            if self.research_mode == "general": # Only generate follow-ups for general research
                queries = list(set(next_cycle_potential_queries))[:max_queries_per_cycle] 
                if not queries:
                    logger.info("✓ All potential follow-up queries have been explored or none were generated.")
                    break
            else: # For market research, we stick to the initial structured queries for now
                queries = [] # Stop after initial structured queries
                break 
            
            time.sleep(1) # Brief pause between cycles

        if not self.report_sections:
            logger.warning("No research sections were generated. Cannot create a report.")
            return "Error: No research content was generated."
            
        return self.compile_and_export_report()

    def process_single_query_for_analysis(self, query: str, cycle: int) -> tuple[str, list]:
        """Processes a single query: web search, then AI analysis. Returns analysis and follow-up questions."""
        logger.info(f"Processing query for analysis: '{query}' (Cycle {cycle + 1})" )
        results = self.web_search(query)
        if not results:
            logger.warning(f"No web results for query: '{query}'. Skipping analysis.")
            return "", []
            
        analysis_text = self.analyze_results(query, results)
        
        follow_ups = []
        if analysis_text:
            # Extract follow-up questions from the end of the analysis_text
            follow_up_match = re.search(r"(?:# Follow-up Research Questions|Follow-up Research Questions:)\s*(.*)", analysis_text, re.DOTALL | re.IGNORECASE)
            if follow_up_match:
                follow_up_block = follow_up_match.group(1).strip()
                # Regex to find lines starting with number, dot, optional space, or bullet, optional space
                raw_queries = re.findall(r"^(?:\d+\.|[-*•])\s*(.+)", follow_up_block, re.MULTILINE)
                follow_ups = [q.strip() for q in raw_queries if q.strip() and len(q.strip()) > 10] # Basic filter for meaningful questions
        
        return analysis_text, follow_ups

    def compile_and_export_report(self) -> str:
        """Compiles all analyzed content into a structured report and saves it."""
        logger.info("\n🖋️ Compiling full research report...")
        if not self.report_sections or not self.initial_topic:
            return "Error: Insufficient data to compile report (no sections or initial topic)."

        report_structure = []
        company_name_for_report = self.initial_topic.split(" and its competitors")[0].strip()

        if self.research_mode == "market_research":
            report_structure = [
                # Abstract is not typical for pure market research, but an Executive Summary is.
                # Let's generate a combined "Executive Summary & Market Outlook" as the intro/conclusion for market research.
                {"title": "Executive Summary & Market Outlook", "type": "Market_Outlook", "source_analysis_key": "combined"}, # Special key
                {"title": f"Market Overview: {company_name_for_report}'s Industry", "type": "Market_Overview", "source_analysis_key": f"Market overview for {company_name_for_report}'s industry"},
                {"title": f"Company Profile: {company_name_for_report}", "type": "Company_Profile", "source_analysis_key": f"Company profile of {company_name_for_report}"},
                {"title": f"Competitive Landscape for {company_name_for_report}", "type": "Competitive_Landscape", "source_analysis_key": f"Competitors of {company_name_for_report}"},
                {"title": f"SWOT Analysis: {company_name_for_report}", "type": "SWOT_Analysis", "source_analysis_key": f"SWOT analysis for {company_name_for_report}"},
                {"title": "Methodology", "type": "Methodology", "source_analysis_key": None}
                # Conclusion is integrated into Market_Outlook for this mode
            ]
        else: # General Research Structure
            report_structure = [
                {"title": "Abstract", "type": "Abstract", "source_analysis_key": None},
                {"title": "Introduction", "type": "Introduction", "source_analysis_key": self.initial_topic},
                {"title": "Methodology", "type": "Methodology", "source_analysis_key": None}
            ]
            for section_data in self.report_sections:
                report_structure.append({"title": section_data['title'], "type": "Results/Findings", "source_analysis_key": section_data['title']})
                report_structure.append({"title": f"Discussion for: {section_data['title']}", "type": "Discussion", "source_analysis_key": section_data['title']})
            report_structure.append({"title": "Conclusion", "type": "Conclusion", "source_analysis_key": None})

        final_report_content = ""
        toc_entries = []

        for section_info in report_structure:
            logger.info(f"Generating section: {section_info['title']} (Type: {section_info['type']})")
            analysis_for_section = ""
            if section_info['source_analysis_key'] == "combined": # For Market Outlook in market research
                # Combine all raw analyses for the Market Outlook section
                all_analyses = [s['raw_analysis'] for s in self.report_sections if s['raw_analysis']]
                analysis_for_section = "\n\n---\n\n".join(all_analyses) if all_analyses else "No detailed analysis available to synthesize outlook."
            elif section_info['source_analysis_key']:
                found_analysis = next((s['raw_analysis'] for s in self.report_sections if s['title'] == section_info['source_analysis_key']), None)
                if found_analysis:
                    analysis_for_section = found_analysis
                elif section_info['type'] == "Introduction": # General research intro fallback
                     analysis_for_section = f"The report covers various sub-topics related to {self.initial_topic}, including: {', '.join(s['title'] for s in self.report_sections)}."
            elif section_info['type'] in ["Abstract", "Conclusion"] and self.research_mode == "general": # General abstract/conclusion themes
                theme_keywords = ", ".join(list(set(s['title'] for s in self.report_sections))[:5])
                analysis_for_section = f"Key research areas explored include: {theme_keywords}. Overall topic: {self.initial_topic}."
            
            section_content = self.generate_report_section_content(
                section_info['title'], 
                analysis_for_section, 
                section_info['type']
            )
            slug = re.sub(r'[^a-zA-Z0-9_\-]', '', section_info['title'].lower().replace(' ', '-'))[:50]
            final_report_content += f"\n\n<a id=\"{slug}\"></a>\n# {section_info['title']}\n\n{section_content}"
            toc_entries.append(f"- [{section_info['title']}](#{slug})")

        toc_md = "## Table of Contents\n" + "\n".join(toc_entries) + "\n"
        references_md = "\n\n# References\n\n"
        if self.reference_map:
            for idx, ref_details in sorted(self.reference_map.items()):
                references_md += f"{idx}. {ref_details['title']} - <{ref_details['url']}>\n"
        else:
            references_md += "No external sources were cited in this report.\n"
        
        report_title = self.initial_topic
        if self.research_mode == "market_research":
            report_title = f"Market Research Report: {company_name_for_report} and its Competitive Landscape"

        full_report_md = f"# {report_title}\n\n{toc_md}\n{final_report_content}\n{references_md}"
        
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = f"research_report_{self.initial_topic.replace(' ', '_').replace('/','_')[:30]}_{timestamp}"
            md_filename = f"{base_filename}.md"
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(full_report_md)
            logger.info(f"\n✅ Markdown Report compiled and saved as: {md_filename} ({len(full_report_md.split())} words approx.)")

            # Save report to memory
            self.memory.add_to_memory(
                "research_reports",
                self.initial_topic,
                {
                    "content_md": full_report_md, # Storing full markdown
                    "timestamp": timestamp,
                    "mode": self.research_mode,
                    "references_count": len(self.reference_map),
                    "search_queries_count": len(self.search_history)
                }
            )
            logger.info(f"Report for '{self.initial_topic}' saved to memory.")
            
            # Attempt PDF export
            self.export_report_to_pdf(full_report_md, base_filename)

            return full_report_md
        except Exception as e:
            logger.error(f"Error saving report to file or memory: {str(e)}")
            return "Error: Could not save the report. Full content might be lost. Check logs."

    def export_report_to_pdf(self, markdown_content: str, base_filename: str):
        """Converts markdown content to HTML and then to PDF, saving it."""
        pdf_filename = f"{base_filename}.pdf"
        try:
            logger.info(f"\n📄 Attempting to export report to PDF: {pdf_filename}...")
            # Basic HTML styling for better PDF output
            html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
            styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Research Report</title>
<style>
    body {{ font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }}
    h1, h2, h3 {{ color: #333; line-height: 1.2; }}
    h1 {{ font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.2em; margin-top: 1.5em; }}
    h3 {{ font-size: 1.2em; margin-top: 1.3em; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }}
    pre {{ background-color: #f8f8f8; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }}
    code {{ font-family: monospace; }}
    a {{ color: #007bff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    #references ul {{ list-style-type: none; padding-left: 0; }}
    #references li {{ margin-bottom: 0.5em; }}
</style>
</head>
<body>
{html_content}
</body>
</html>"""
            
            html_doc = WeasyHTML(string=styled_html)
            html_doc.write_pdf(pdf_filename)
            logger.info(f"✅ PDF Report successfully saved as: {pdf_filename}")
        except Exception as e:
            logger.error(f"❌ Error exporting report to PDF: {str(e)}")
            logger.info("Please ensure WeasyPrint and its dependencies (like Pango, Cairo) are correctly installed.")
            logger.info("For Debian/Ubuntu, try: sudo apt-get install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0")

    def recall_past_research_reports(self, keywords: list[str], similarity_threshold: int = 1) -> list[dict]:
        """
        Searches memory for research reports matching a list of keywords.

        Args:
            keywords (list[str]): A list of keywords to search for in report topics.
            similarity_threshold (int): The minimum number of keywords that must match
                                        for a report to be considered relevant. Defaults to 1.

        Returns:
            list[dict]: A list of matching reports, each containing topic, timestamp,
                        and a preview of the content.
        """
        logger.info(f"Recalling past research reports with keywords: {keywords} (threshold: {similarity_threshold})")
        all_report_topics = self.memory.list_memory_type("research_reports")
        matching_reports = []

        if not all_report_topics:
            logger.info("No research reports found in memory.")
            return []

        for topic in all_report_topics:
            report_data = self.memory.get_from_memory("research_reports", topic)
            if report_data and isinstance(report_data, dict):
                match_count = 0
                for keyword in keywords:
                    if keyword.lower() in topic.lower():
                        match_count += 1

                if match_count >= similarity_threshold:
                    content_preview = report_data.get("content_md", "")[:200] + "..." if report_data.get("content_md") else "No content preview available."
                    matching_reports.append({
                        "topic": topic,
                        "timestamp": report_data.get("timestamp", "N/A"),
                        "mode": report_data.get("mode", "N/A"),
                        "content_preview": content_preview,
                        "references_count": report_data.get("references_count", 0),
                        "search_queries_count": report_data.get("search_queries_count", 0)
                    })

        if matching_reports:
            logger.info(f"Found {len(matching_reports)} relevant reports.")
        else:
            logger.info("No relevant reports found matching the criteria.")
        return matching_reports

    def list_all_research_topics(self) -> list[str]:
        """
        Lists all research topics for which reports are stored in memory.

        Returns:
            list[str]: A list of all research topics (keys) from the "research_reports" memory type.
        """
        logger.info("Listing all research topics from memory.")
        topics = self.memory.list_memory_type("research_reports")
        if topics:
            logger.info(f"Found {len(topics)} research topics in memory.")
        else:
            logger.info("No research topics found in memory.")
        return topics

    def get_specific_report(self, topic: str) -> dict | None:
        """
        Retrieves a specific research report from memory by its topic.

        Args:
            topic (str): The exact topic of the report to retrieve.

        Returns:
            dict | None: The report data (dictionary) if found, otherwise None.
        """
        logger.info(f"Retrieving specific report for topic: '{topic}'")
        report_data = self.memory.get_from_memory("research_reports", topic)
        if report_data:
            logger.info(f"Report found for topic '{topic}'.")
        else:
            logger.warning(f"No report found for topic '{topic}'.")
        return report_data

# Example usage:
if __name__ == "__main__":
    agent = ResearchAgent()
    
    # Switch to General Research on GRPO with Unsloth
    logger.info("\n--- Starting General Research for In-depth Report --- ")
    general_topic = "Investigate why major AI labs are not widely adopting or publishing on memory-augmented architectures despite their biological inspiration, potential for self-directed learning, and long-term reasoning capabilities."
    # Check memory for this topic before running
    existing_report_data = agent.memory.get_from_memory("research_reports", general_topic)
    if existing_report_data:
        logger.info(f"Found existing report for '{general_topic}' in memory from {existing_report_data.get('timestamp', 'N/A')}.")
        # Example: Could add a prompt here: "Do you want to re-run research or view existing?"
        # For now, we'll just proceed to re-run. A real application might offer choices.

    final_report_markdown_general = agent.research_cycle(
        topic=general_topic,
        depth=3, # Increase depth for more comprehensive coverage (3-4 recommended)
        research_mode="general",
        per_query_delay=10 # Increased delay between processing queries
    )
    if isinstance(final_report_markdown_general, str) and not final_report_markdown_general.startswith("Error"):
        # The PDF export is now called within compile_and_export_report
        # No need to call it separately here if successful.
        logger.info(f"\n--- GENERAL REPORT (Markdown) PREVIEW ---")
        # Limiting preview to avoid excessive console output
        # The full report is saved to MD and PDF files.
        # print(final_report_markdown_general[:3000] + "...") # For brevity in example
    else:
        logger.error(f"\nGeneral research failed or produced no report: {final_report_markdown_general}")

    # Market Research Example (can be uncommented to run)
    # logger.info("\n--- Starting Market Research --- ")
    # market_research_topic = "Tesla and its competitors"
    # # Quick check if a report on Tesla already exists
    # tesla_reports = agent.recall_past_research_reports(keywords=["Tesla"], similarity_threshold=1)
    # if tesla_reports:
    #     logger.info(f"Found {len(tesla_reports)} existing reports that might be related to 'Tesla':")
    #     for r_info in tesla_reports:
    #         logger.info(f"  - Topic: {r_info['topic']} (Timestamp: {r_info['timestamp']})")
    #
    # final_report_markdown_market = agent.research_cycle(
    #     topic=market_research_topic,
    #     research_mode="market_research"
    # )
    # if isinstance(final_report_markdown_market, str) and not final_report_markdown_market.startswith("Error"):
    #     logger.info(f"\n--- MARKET RESEARCH REPORT PREVIEW ---")
    #     # print(final_report_markdown_market[:2000] + "...")
    # else:
    #     logger.error(f"\nMarket research failed or produced no report: {final_report_markdown_market}")

    logger.info("\n--- Demonstrating Memory Interaction Methods ---")
    all_topics = agent.list_all_research_topics()
    logger.info(f"All research topics currently in memory: {all_topics}")

    if all_topics:
        logger.info(f"\nAttempting to recall reports related to 'AI' and 'memory':")
        recalled_reports_ai_memory = agent.recall_past_research_reports(keywords=["AI", "memory"], similarity_threshold=2)
        if recalled_reports_ai_memory:
            for report_info in recalled_reports_ai_memory:
                logger.info(f"  Found matching report: '{report_info['topic']}' (Mode: {report_info['mode']}, Timestamp: {report_info['timestamp']})")
                logger.info(f"    Preview: {report_info['content_preview']}")
        else:
            logger.info("  No reports found matching 'AI' and 'memory' with threshold 2.")

        logger.info(f"\nAttempting to recall reports related to 'biological':")
        recalled_reports_bio = agent.recall_past_research_reports(keywords=["biological"])
        if recalled_reports_bio:
            for report_info in recalled_reports_bio:
                logger.info(f"  Found matching report: '{report_info['topic']}' (Mode: {report_info['mode']}, Timestamp: {report_info['timestamp']})")
        else:
            logger.info("  No reports found matching 'biological'.")

        # Get a specific report (assuming one of the all_topics exists)
        if general_topic in all_topics: # Use the topic from the earlier general research run
            logger.info(f"\nAttempting to retrieve specific report for topic: '{general_topic}'")
            specific_report = agent.get_specific_report(general_topic)
            if specific_report:
                logger.info(f"Successfully retrieved report for '{general_topic}'. Timestamp: {specific_report.get('timestamp')}")
                # logger.info(f"Full content of '{general_topic}':\n{specific_report.get('content_md')[:500]}...") # Potentially very long
            else:
                logger.info(f"Could not retrieve report for '{general_topic}' (this shouldn't happen if it was just created).")
    else:
        logger.info("No topics in memory to demonstrate recall or specific get.")

    logger.info("\n--- Demonstrating Shared Memory with PDFVisionAssistant ---")
    # Instantiate PDFVisionAssistant, passing the ResearchAgent's memory instance
    # This allows PDFVisionAssistant to save its findings to the same memory file.
    pdf_assistant = PDFVisionAssistant(memory_instance=agent.memory)

    # Simulate PDFVisionAssistant processing a PDF and saving its extraction to memory
    # In a real scenario, you would call:
    # pdf_assistant.extract_text_from_pdf("path/to/actual/document.pdf")
    # or pdf_assistant.analyze_pdf_layout_and_text("path/to/actual/document.pdf")

    simulated_pdf_filename = "annual_report_2023.pdf" # Using basename as key, as per PDFVisionAssistant's implementation
    simulated_pdf_full_path = f"dummy/path/to/{simulated_pdf_filename}" # Full path for context in the value
    timestamp_pdf = time.strftime("%Y%m%d-%H%M%S")

    # Manually add a simulated entry as if PDFVisionAssistant saved it
    agent.memory.add_to_memory(
        memory_type="pdf_extractions",
        key=simulated_pdf_filename, # PDFVisionAssistant uses os.path.basename(pdf_path) as key
        value={
            "type": "text",
            "content": "This is the simulated extracted text from the annual_report_2023.pdf. It contains important financial data and strategic outlooks.",
            "timestamp": timestamp_pdf,
            "source_pdf_path": simulated_pdf_full_path
        }
    )
    logger.info(f"Simulated saving of PDF extraction for '{simulated_pdf_filename}' to shared memory by PDFVisionAssistant.")

    # Now, demonstrate ResearchAgent retrieving this data
    retrieved_pdf_data = agent.memory.get_from_memory("pdf_extractions", simulated_pdf_filename)
    if retrieved_pdf_data:
        logger.info(f"ResearchAgent found extracted PDF data for '{simulated_pdf_filename}' in shared memory.")
        logger.info(f"  Retrieved data type: {retrieved_pdf_data.get('type')}")
        logger.info(f"  Retrieved content snippet: {retrieved_pdf_data.get('content', '')[:50]}...")
        logger.info(f"  Retrieved timestamp: {retrieved_pdf_data.get('timestamp')}")
        logger.info(f"  Original PDF path: {retrieved_pdf_data.get('source_pdf_path')}")
    else:
        logger.info(f"ResearchAgent did not find PDF data for '{simulated_pdf_filename}' in shared memory (this should not happen in simulation).")

    # Example of how ResearchAgent might list all PDF extractions
    all_pdf_extractions_keys = agent.memory.list_memory_type("pdf_extractions")
    logger.info(f"\nAll PDF extraction keys found in memory by ResearchAgent: {all_pdf_extractions_keys}")
    if simulated_pdf_filename in all_pdf_extractions_keys:
        logger.info(f"Confirmed '{simulated_pdf_filename}' is listed under 'pdf_extractions'.")
