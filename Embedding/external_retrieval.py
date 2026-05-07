import requests
import json
import time
from bs4 import BeautifulSoup
try:
    import trafilatura
except ImportError:
    trafilatura = None

# A simple in-memory session cache to avoid repeated external calls during the demo
_SEARCH_CACHE = {}

def fetch_indian_kanoon_doc(doc_id: str, api_token: str = None) -> str:
    """Fetches full text for an Indian Kanoon document given its ID."""
    if not doc_id:
        return ""
        
    url = f"https://api.indiankanoon.org/doc/{doc_id}/"
    headers = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Token {api_token}"
        
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            raw_content = data.get("doc", "")
            
            # Clean HTML using Trafilatura if available, else BeautifulSoup
            if trafilatura:
                return trafilatura.extract(raw_content) or ""
            else:
                soup = BeautifulSoup(raw_content, "html.parser")
                return soup.get_text(separator=" ", strip=True)
        return ""
    except Exception as e:
        print(f"Error fetching Kanoon doc {doc_id}: {e}")
        return ""

def search_legal_context(query: str, limit: int = 1) -> list[dict]:
    """
    Performs a single-shot search of legal context for the entire query.
    Note: For hackathons, we can also fallback to a pre-defined 'Golden Knowledge Base' in Mongo.
    """
    if not query:
        return []
    
    if query in _SEARCH_CACHE:
        return _SEARCH_CACHE[query]
        
    results = []
    
    # --- Indian Kanoon Endpoint (Mocked Search logic for demonstration) ---
    # In a real scenario, we'd use api.indiankanoon.org/search/
    # For now, we provide the structure for the Orchestrator to plug into.
    
    print(f"Searching external legal context for query: '{query[:50]}...'")
    
    # (Placeholder for real API search loop)
    # response = requests.get(f"https://api.indiankanoon.org/search/?formInput={query}", ...)
    
    # Cache the result to prevent redundant calls
    _SEARCH_CACHE[query] = results
    return results

def scrape_sebi_circulars(keyword: str):
    """Fallback scraper for SEBI circulars listing."""
    url = f"https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes&cid=1"
    # Note: Modern websites often require headers to avoid generic bots blocks
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract circular links based on keyword (simple filter)
            links = []
            for a in soup.find_all("a", href=True):
                if keyword.lower() in a.get_text().lower():
                    links.append({"title": a.get_text().strip(), "url": a["href"]})
            return links[:3]
        return []
    except Exception as e:
        print(f"SEBI scraping error: {e}")
        return []

if __name__ == "__main__":
    print("Testing external retrieval components...")
    res = scrape_sebi_circulars("Margin")
    print(f"Found {len(res)} matching SEBI circulars.")
    for r in res:
        print(f"- {r['title']}")
