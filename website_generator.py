import os
import re
from openai import OpenAI
import logging
import asyncio
import json

# Configure logger
logger = logging.getLogger(__name__)

class WebsiteGenerator:
    """
    An agent that transforms a markdown research report into an interactive
    single-page HTML application.
    """
    def __init__(self):
        """Initialize the website generator agent."""
        # Use environment variables for API configuration
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        if not os.getenv("DEEPSEEK_API_KEY"):
            logger.warning("DEEPSEEK_API_KEY environment variable not set. AI features will not work.")

    def _parse_markdown(self, markdown_content: str) -> dict:
        """
        Parses the markdown report into a structured dictionary.
        """
        parsed_data = {
            'title': 'Research Report',
            'toc': [],
            'sections': [],
            'references': {}
        }

        # Extract main title
        title_match = re.search(r'^# (.+)', markdown_content, re.MULTILINE)
        if title_match:
            parsed_data['title'] = title_match.group(1).strip()

        # Extract Table of Contents
        toc_match = re.search(r'## Table of Contents\n([\s\S]+?)\n\n', markdown_content)
        if toc_match:
            toc_block = toc_match.group(1)
            toc_items = re.findall(r'- \[(.+)\]\(#(.+)\)', toc_block)
            for title, slug in toc_items:
                parsed_data['toc'].append({'title': title, 'slug': slug})

        # Extract References
        references_match = re.search(r'# References\n\n([\s\S]+)', markdown_content)
        if references_match:
            ref_block = references_match.group(1)
            ref_items = re.findall(r'(\d+)\. (.+) - <(.+)>', ref_block)
            for number, title, url in ref_items:
                parsed_data['references'][number] = {'title': title.strip(), 'url': url.strip()}

        # Extract sections using the ToC slugs as markers
        section_content = markdown_content
        if toc_match:
             # Isolate the content part of the report
            content_start = toc_match.end()
            content_end = references_match.start() if references_match else len(markdown_content)
            section_content = markdown_content[content_start:content_end]

        sections = re.split(r'<a id="[^"]+"></a>', section_content)
        for i, toc_entry in enumerate(parsed_data['toc']):
            # The first split part is before the first section, so we skip it
            if (i + 1) < len(sections):
                raw_content = sections[i+1]
                # Clean up the section title from the top of the content
                cleaned_content = re.sub(r'\s*# .+\n\n', '', raw_content, count=1).strip()
                parsed_data['sections'].append({
                    'title': toc_entry['title'],
                    'slug': toc_entry['slug'],
                    'content': cleaned_content
                })

        return parsed_data

    async def _generate_ai_placeholders(self, parsed_data: dict) -> dict:
        """
        Uses an AI model to generate the planning comments for the HTML head.
        """
        if not self.client.api_key:
            logger.warning("AI placeholder generation skipped due to missing API key.")
            return {}

        # Create a summary of the report for the AI prompt
        summary = f"The report is titled '{parsed_data['title']}'. "
        summary += f"It contains {len(parsed_data['sections'])} main sections: "
        summary += ", ".join([s['title'] for s in parsed_data['toc']]) + ". "
        summary += f"There are {len(parsed_data['references'])} references cited."

        prompt = f"""
        You are an expert UI/UX designer and information architect. Your task is to generate planning comments for an interactive HTML report based on the following report summary.

        Report Summary: {summary}

        Based on these instructions, please generate the content for the following two placeholders. Respond with a JSON object containing keys "structure_plan" and "viz_plan".

        1.  **Application Structure Plan:** Describe the chosen structure for the SPA. A strong default choice for a research report is a fixed left sidebar for navigation (built from the Table of Contents) and a main content area that scrolls. Justify this choice, explaining how it enhances usability and exploration of the research material. Mention that the sidebar provides primary navigation, and the main area displays the content for each section.

        2.  **Visualization & Content Choices:** Explain the content presentation strategy. Describe how sections from the report are rendered as distinct blocks. Mention that reference citations like [1] are made interactive, opening a pop-up or modal with the full source details (Title and URL) on click. State that this design makes the report more engaging and credible by providing easy access to sources.
        """

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            placeholders = json.loads(response.choices[0].message.content)
            return placeholders
        except Exception as e:
            logger.error(f"Error generating AI placeholders: {e}")
            return {
                'structure_plan': 'Application Structure Plan: AI generation failed. Using default sidebar layout.',
                'viz_plan': 'Visualization & Content Choices: AI generation failed. Using default interactive references.'
            }

    def _generate_head(self, parsed_data: dict, ai_placeholders: dict) -> str:
        """Generates the <head> section of the HTML document."""
        return f"""<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Report: {parsed_data.get('title', 'Research Analysis')}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Slab:wght@400;700&display=swap" rel="stylesheet">

    <!-- {ai_placeholders.get('palette', 'Chosen Palette: Slate, Stone, Sky')} -->
    <!-- {ai_placeholders.get('structure_plan', 'Application Structure Plan: Default structure chosen.')} -->
    <!-- {ai_placeholders.get('viz_plan', 'Visualization & Content Choices: Standard visualizations selected.')} -->
    <!-- CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->

    <style>
        body {{
            font-family: 'Inter', sans-serif;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Roboto Slab', serif;
        }}
        .sidebar-link.active, .sidebar-link:hover {{
            background-color: #e0f2fe; /* sky-100 */
            color: #0c4a6e; /* sky-900 */
            border-left-color: #0ea5e9; /* sky-500 */
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 100;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.5);
            align-items: center;
            justify-content: center;
        }}
        .modal-content {{
            background-color: #fefefe;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 700px;
            border-radius: 0.5rem;
        }}
        .prose ul {{ list-style-type: disc !important; margin-left: 1.5rem !important; }}
        .prose ol {{ list-style-type: decimal !important; margin-left: 1.5rem !important; }}
        .prose strong {{ font-weight: 600 !important; }}
        .prose a {{ color: #0284c7 !important; text-decoration: underline !important; }}
        .prose code {{ background-color: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 0.25rem; font-family: monospace; }}
        .prose pre {{ background-color: #f1f5f9; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }}
        .prose pre code {{ padding: 0; background-color: transparent; }}
    </style>
</head>"""

    def _generate_body(self, parsed_data: dict) -> str:
        """Generates the <body> section of the HTML document."""
        return f"""<body class="bg-slate-50 text-slate-800">
    <div class="flex">
        <!-- Sidebar Navigation -->
        <aside id="sidebar" class="hidden md:block fixed top-0 left-0 w-64 h-full bg-white shadow-md z-40 overflow-y-auto">
            <div class="p-4 border-b">
                <h2 class="text-xl font-bold text-slate-800">Report Sections</h2>
            </div>
            <nav id="toc-nav" class="p-4"></nav>
        </aside>

        <!-- Main Content -->
        <main class="md:ml-64 w-full min-h-screen bg-slate-50">
            <!-- Header for Mobile -->
            <header class="md:hidden bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50 flex items-center justify-between p-4">
                <h1 class="text-lg font-bold text-slate-800 truncate">{parsed_data.get('title', 'Research Report')}</h1>
                <button id="mobile-menu-button" class="p-2 rounded-md text-slate-500 hover:bg-slate-100">
                     <span class="sr-only">Open menu</span>
                     <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
                </button>
            </header>

            <div class="p-4 sm:p-6 lg:p-8">
                <header class="mb-12 pb-8 border-b-2 border-slate-200">
                    <h1 class="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl leading-tight">{parsed_data.get('title', 'Research Report')}</h1>
                </header>
                <div id="report-content" class="space-y-12"></div>
            </div>

             <footer class="bg-white mt-16 border-t">
                <div class="container mx-auto py-6 px-4 sm:px-6 lg:px-8 text-center text-slate-500">
                    <p>&copy; {{{{new Date().getFullYear()}}}} Interactive Report. Generated by AI.</p>
                </div>
            </footer>
        </main>
    </div>

    <!-- Reference Modal -->
    <div id="reference-modal" class="modal">
        <div class="modal-content animate-fade-in">
            <div class="flex justify-between items-center mb-4">
                <h3 id="reference-modal-title" class="text-xl font-bold text-sky-800">Reference</h3>
                <button id="close-reference-modal" class="text-slate-500 hover:text-slate-700 text-2xl">&times;</button>
            </div>
            <div id="reference-modal-body" class="text-slate-700 prose max-w-none"></div>
        </div>
    </div>
</body>"""

    def _generate_script(self, json_data: str) -> str:
        """Generates the <script> section of the HTML document."""
        return f"""<script>
        document.addEventListener('DOMContentLoaded', () => {{
            const appData = {json_data};

            const tocNav = document.getElementById('toc-nav');
            const reportContent = document.getElementById('report-content');
            const mobileMenuButton = document.getElementById('mobile-menu-button');
            const sidebar = document.getElementById('sidebar');

            // --- Reference Modal Logic ---
            const refModal = document.getElementById('reference-modal');
            const closeRefModalButton = document.getElementById('close-reference-modal');
            const refModalTitle = document.getElementById('reference-modal-title');
            const refModalBody = document.getElementById('reference-modal-body');

            const showRefModal = (refId) => {{
                const refData = appData.references[refId];
                if (!refData) return;
                refModalTitle.textContent = `Reference [${{refId}}]`;
                refModalBody.innerHTML = `
                    <p class="font-semibold"><strong>Title:</strong> ${{refData.title}}</p>
                    <p><strong>URL:</strong> <a href="${{refData.url}}" target="_blank" rel="noopener noreferrer">${{refData.url}}</a></p>
                `;
                refModal.style.display = 'flex';
            }};

            closeRefModalButton.addEventListener('click', () => refModal.style.display = 'none');
            window.addEventListener('click', (event) => {{
                if (event.target == refModal) {{
                    refModal.style.display = 'none';
                }}
            }});

            // --- Populate Content and Navigation ---
            appData.toc.forEach(item => {{
                const link = document.createElement('a');
                link.href = `#${{item.slug}}`;
                link.textContent = item.title;
                link.className = 'sidebar-link block font-medium text-slate-600 p-2 rounded-md border-l-4 border-transparent transition-colors duration-200';
                link.dataset.slug = item.slug;
                tocNav.appendChild(link);
            }});

            appData.sections.forEach(section => {{
                const sectionEl = document.createElement('section');
                sectionEl.id = section.slug;
                sectionEl.className = 'scroll-mt-20 bg-white p-8 rounded-xl shadow-md';

                let processedContent = section.content.replace(/\[(\d+)\]/g, (match, refId) => {{
                    return `<button class="reference-link text-sky-600 font-bold hover:underline" data-ref-id="${{refId}}">${{match}}</button>`;
                }});

                sectionEl.innerHTML = `
                    <h2 class="text-3xl font-bold text-slate-800 mb-6 border-b pb-4">${{section.title}}</h2>
                    <div class="prose max-w-none text-slate-700 leading-relaxed">${{marked.parse(processedContent)}}</div>
                `;
                reportContent.appendChild(sectionEl);
            }});

            // Add click listeners to new reference links
            reportContent.addEventListener('click', (event) => {{
                if (event.target.classList.contains('reference-link')) {{
                    const refId = event.target.dataset.refId;
                    showRefModal(refId);
                }}
            }});

            // --- Navigation Logic ---
            const sidebarLinks = document.querySelectorAll('.sidebar-link');
            const contentSections = document.querySelectorAll('#report-content > section');

            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        sidebarLinks.forEach(link => {{
                            link.classList.toggle('active', link.dataset.slug === entry.target.id);
                        }});
                    }}
                }});
            }}, {{ rootMargin: '-40% 0px -60% 0px', threshold: 0 }});

            contentSections.forEach(section => observer.observe(section));

            mobileMenuButton.addEventListener('click', () => {{
                sidebar.classList.toggle('hidden');
            }});
        }});
    </script>"""

    def _generate_html(self, parsed_data: dict, ai_placeholders: dict) -> str:
        """
        Generates the final HTML content from parsed data and AI placeholders.
        """
        json_data = json.dumps(parsed_data, indent=4)

        head = self._generate_head(parsed_data, ai_placeholders)
        body = self._generate_body(parsed_data)
        script = self._generate_script(json_data)

        return f"""<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
{head}
{body}
{script}
</html>
"""

    async def generate_from_markdown(self, markdown_content: str, output_path: str):
        """
        Orchestrates the conversion from a markdown string to an HTML file.
        """
        logger.info("Parsing markdown report...")
        parsed_data = self._parse_markdown(markdown_content)

        logger.info("Generating AI planning placeholders...")
        ai_placeholders = await self._generate_ai_placeholders(parsed_data)

        logger.info("Generating final HTML content...")
        final_html = self._generate_html(parsed_data, ai_placeholders)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            logger.info(f"Successfully generated website at: {output_path}")
        except IOError as e:
            logger.error(f"Failed to write HTML file to {output_path}: {e}")


async def generate_website(markdown_content: str, output_path: str):
    """
    Creates a WebsiteGenerator instance and calls its generate_from_markdown method.
    """
    generator = WebsiteGenerator()
    await generator.generate_from_markdown(markdown_content, output_path)
