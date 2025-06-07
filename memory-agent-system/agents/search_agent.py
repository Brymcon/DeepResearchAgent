import os
import json
import requests # For making HTTP requests to a search API

class SearchAgent:
    def __init__(self):
        # In a real scenario, API keys and endpoints would be loaded from config
        # For example, os.environ.get('SERPAPI_KEY')
        self.api_key = os.environ.get('SEARCH_API_KEY') # Generic name
        self.search_engine_url = os.environ.get('SEARCH_ENGINE_URL') # e.g., SerpAPI endpoint or Google PSE URL
        print("SearchAgent: Initialized.")
        if not self.api_key or not self.search_engine_url:
            print("SearchAgent: WARNING - API key or Search Engine URL not configured. Search will use mock results.")

    def search(self, query: str, num_results: int = 3) -> list[dict]:
        """
        Performs a web search for the given query.

        Args:
            query: The search query string.
            num_results: The desired number of search results.

        Returns:
            A list of dictionaries, where each dictionary represents a search result
            and contains keys like 'title', 'link', 'snippet'.
            Returns mock results if API key/URL is not available.
        """
        print(f"SearchAgent: Searching for '{query}'...")

        if not self.api_key or not self.search_engine_url:
            print("SearchAgent: Using mock search results as API key/URL is not available.")
            return self._get_mock_results(query, num_results)

        # This is a generic structure; actual parameters and headers will vary by search API
        # Example for a hypothetical API, similar to how SerpAPI might work
        params = {
            'q': query,
            'api_key': self.api_key,
            'num': num_results
            # Other parameters like 'location', 'hl' (language) might be needed
        }
        headers = {
            'Accept': 'application/json'
        }

        try:
            response = requests.get(self.search_engine_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            search_data = response.json()
            return self._parse_results(search_data, num_results)
        except requests.exceptions.RequestException as e:
            print(f"SearchAgent: API request failed: {e}")
            return self._get_mock_results(query, num_results, error_occurred=True)
        except json.JSONDecodeError as e:
            print(f"SearchAgent: Failed to decode API response: {e}")
            return self._get_mock_results(query, num_results, error_occurred=True)

    def _parse_results(self, search_data: dict, num_results: int) -> list[dict]:
        """Parses the JSON response from the search API into a standard format."""
        results = []
        # This parsing logic is highly dependent on the actual API's response structure
        # Example: for SerpAPI, results are often in 'organic_results'
        # For Google PSE, it might be in 'items'
        if 'organic_results' in search_data:
            api_results = search_data['organic_results']
        elif 'items' in search_data: # Common in Google Custom Search JSON API
            api_results = search_data['items']
        else:
            api_results = []

        for item in api_results[:num_results]:
            results.append({
                'title': item.get('title', 'No title available'),
                'link': item.get('link', '#'),
                'snippet': item.get('snippet', 'No snippet available')
            })
        if not results and api_results: # If parsing failed but there were items
             print(f"SearchAgent: Could not parse API results into standard format. Raw items: {len(api_results)}")
        elif not api_results:
             print("SearchAgent: No results found in API response.")
        return results

    def _get_mock_results(self, query: str, num_results: int, error_occurred: bool = False) -> list[dict]:
        """Generates mock search results for when the API is not available."""
        if error_occurred:
            prefix = "Error-Mock"
        else:
            prefix = "Mock"
        mock_results = []
        for i in range(num_results):
            mock_results.append({
                'title': f"{prefix} Title {i+1} for '{query}'",
                'link': f"http://example.com/mock_search_result_{i+1}",
                'snippet': f"{prefix} snippet for search result {i+1} related to query '{query}'. This is placeholder content."
            })
        return mock_results

# Example usage (for testing this agent directly):
# if __name__ == '__main__':
#     search_agent = SearchAgent()
#     print("--- Testing with Mock Results (default) ---")
#     results = search_agent.search("what is AI?")
#     for r in results:
#         print(f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}\n---")

#     print("\n--- Simulating API Key for potential live test (requires actual key & URL in env) ---")
#     # To test live, set SEARCH_API_KEY and SEARCH_ENGINE_URL environment variables
#     # For example, for SerpAPI, SEARCH_ENGINE_URL might be "https://serpapi.com/search"
#     if os.environ.get('SEARCH_API_KEY') and os.environ.get('SEARCH_ENGINE_URL'):
#         print("Attempting live search...")
#         live_results = search_agent.search("latest AI news", num_results=2)
#         for r in live_results:
#             print(f"Title: {r['title']}\nLink: {r['link']}\nSnippet: {r['snippet']}\n---")
#     else:
#         print("Skipping live test as API key/URL not found in environment.")
