from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import requests
import re
from google import genai
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import tldextract
import scholarly

load_dotenv()

key = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=key)

# --- Pydantic Models for Input Validation and Documentation ---

class LinkSelector(BaseModel):
    """Model for defining how to find profile links on the index page."""
    tag: str = Field(..., description="HTML tag of the link element (e.g., 'a').")
    # Note: Using Optional[str] allows 'class' to be omitted if not needed.
    class_: Optional[str] = Field(None, alias="class", description="CSS class of the link element (e.g., 'profile-link').")

class DetailSelectors(BaseModel):
    """Model for defining how to extract data from the individual profile page."""
    name_tag: str = Field(..., description="HTML tag for the person's name (e.g., 'h1').")
    name_class: Optional[str] = Field(None, alias="name_class", description="CSS class for the person's name (e.g., 'person-name').")
    
    main_text_tag: str = Field(..., description="HTML tag for the main content block (e.g., 'div').")
    main_text_class: Optional[str] = Field(None, alias="main_text_class", description="CSS class for the main content block (e.g., 'profile-content-body').")

class ScrapingConfig(BaseModel):
    """Model representing the full configuration for scraping a single website."""
    site_name: str = Field(..., description="A friendly name for the site being scraped.")
    index_url: HttpUrl = Field(..., description="The URL of the index page listing all profiles.")
    
    link_selector: LinkSelector
    detail_selectors: DetailSelectors
    
    # Allows use of field names like 'class' and 'class_'
    class Config:
        allow_population_by_field_name = True

class ProfileData(BaseModel):
    """Model for the returned scraped data structure."""
    site: str
    url: str
    name: str
    full_text: str
    
class ScrapeRequest(BaseModel):
    url: str
    selector: str
    
# API Routes

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    # You might need to add the actual domain/IP where your client is hosted
    "*" # Using '*' for maximum flexibility in this demo environment
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/scrape", response_model=List[str])
async def scrape_website(request_data: ScrapeRequest):
    """
    Fetches the given URL and extracts text content using the provided CSS selector.
    """
    url = request_data.url
    selector = request_data.selector
    
    if not url or not selector:
        raise HTTPException(status_code=400, detail="URL and selector must be provided.")

    try:
        # Use a common user-agent header to avoid being blocked by simple checks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 4. Fetch the content using requests (the magic that bypasses CORS!)
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # 5. Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 6. Find all elements matching the selector
        elements = soup.select(selector)

        if not elements:
            # Return an empty list if no elements are found
            return []

        names = []
        for el in elements:
            # Extract text content, clean up whitespace, and remove excessive newlines/tabs
            text = el.get_text(strip=True)
            if text:
                # Further cleanup: replace multiple spaces with a single space
                cleaned_text = re.sub(r'\s+', ' ', text)
                names.append(cleaned_text)
                
        domain_name = tldextract.extract(url).domain
        author_info = []
        for name in names:
            search_query = scholarly.search_author(f'{name}, {domain_name}')
            author = next(search_query)
            scholarly.fill(author, sections=['basics', 'coauthors'])
            author_info.append(author)

        return author_info

    except requests.exceptions.RequestException as e:
        # Handle network-related errors (DNS failure, connection refused, timeout)
        raise HTTPException(status_code=502, detail=f"Failed to fetch external URL: {e}")
    except Exception as e:
        # Handle all other parsing or unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal error occurred during scraping: {e}")

@app.get("/health")
def health_check():
    return {'status': 'healthy'}