from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import requests
import re
from google import genai
from google.genai import types
from google.genai.errors import APIError 

from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import time
import json

# NEW: Imports for Clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # Using PCA for 2D coordinate generation
from sklearn.exceptions import ConvergenceWarning
import warnings

import redis
import base64

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)

SCEMANTIC_SCHOLAR_API = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

N_CLUSTERS = 3

CACHE_EXPIRATION_SECONDS = 7200

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# --- Pydantic Models for Input Validation and Documentation ---
    
class ScrapeRequest(BaseModel):
    url: str
    selector: str
    
class ResearchResult(BaseModel):
    name: str
    research_summary: str
    sources: List[Dict[str, str]]
    # Field to hold the assigned cluster ID
    cluster_id: Optional[int] = None 
    # Field to hold the common co-authors
    common_co_authors: Optional[str] = None
    # NEW: 2D coordinates for the relationship map visualization (from PCA)
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    
def generate_embedding(text: str, client: genai.Client) -> Optional[List[float]]:
    """Generates an embedding vector for a given text using text-embedding-004."""
    try:
        # Use a reliable text embedding model
        response = client.models.embed_content(
            model='text-embedding-004',
            content=text,
            task_type='CLUSTERING',
        )
        return response.embedding
    except APIError as e:
        print(f"Embedding API Error for text: {text[:50]}... Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Embedding Error: {e}")
        return None
    
    
def find_research_papers(name: str, client: genai.Client) -> ResearchResult:
    """
    Calls the Gemini API using the official Python SDK with Google Search grounding.
    """
    
    # Prompting for co-authors and summary
    prompt = (
        f"Find the three most important or recent research papers published by the academic or professional named '{name}'. "
        f"Also, identify their 3 most common co-authors (or known collaborators). "
        f"Format the response such that the first line contains ONLY the co-authors, prefixed by 'COAUTHORS:', separated by commas. "
        f"The subsequent lines should contain the research summary (title, year, and focus) formatted as a concise list or paragraph."
    )

    # Define configuration for the API call
    config = types.GenerateContentConfig(
        tools=[{"google_search": {}}],
        system_instruction="You are a research assistant. Provide the response exactly as requested. List the co-authors on the first line separated by commas, and then provide the research summary. State clearly if no information is found for either part."
    )
    
    # Create the contents list
    contents = [types.Content(parts=[types.Part.from_text(text=prompt)])]

    try:
        # Call the Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=contents,
            config=config
        )

        # Parse the response text
        text = response.text or 'No text generated.'
        
        # Attempt to parse co-authors and summary based on the requested format
        co_authors = None
        summary_text = text
        
        if text.lower().startswith("coauthors:"):
            lines = text.split('\n', 1)
            co_author_line = lines[0]
            
            # Simple extraction after the prefix (case-insensitive)
            match = re.search(r'coauthors:(.*)', co_author_line, re.IGNORECASE)
            if match:
                co_authors = match.group(1).strip()

            # The rest of the text is the summary
            summary_text = lines[1].strip() if len(lines) > 1 else "No detailed summary provided."
        
        # Extract grounding sources from the SDK response object
        sources = []
        if response.candidates and response.candidates[0].grounding_metadata:
            grounding_metadata = response.candidates[0].grounding_metadata
            
            if grounding_metadata.grounding_attributions:
                sources = [
                    {"uri": attr.web.uri, "title": attr.web.title}
                    for attr in grounding_metadata.grounding_attributions
                    if attr.web and attr.web.uri and attr.web.title
                ]
        
        return ResearchResult(
            name=name,
            research_summary=summary_text,
            sources=sources,
            common_co_authors=co_authors
        )

    except APIError as e:
        # Handle API specific errors (e.g., rate limits, invalid key)
        return ResearchResult(
            name=name,
            research_summary=f"Gemini API Error (Check API Key and Rate Limits): {e}",
            sources=[],
            cluster_id=-2 # Special ID for API failure
        )
    except Exception as e:
        # Handle general unexpected errors
        return ResearchResult(
            name=name,
            research_summary=f"An unexpected error occurred during Gemini call: {e}",
            sources=[],
        )
    
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

    # --- Caching Check ---
    cache_key = f"research_map:{url}:{selector}"
    cached_results_json = r.get(cache_key)
    if cached_results_json:
        print("Cache hit! Returning cached results.")
        # Deserialize JSON back into Pydantic models
        try:
            cached_data = json.loads(cached_results_json)
            return [ResearchResult(**data) for data in cached_data]
        except Exception as e:
            # If deserialization fails (e.g., cache data is corrupted or structure changed), re-run
            print(f"Error deserializing cached data: {e}. Re-running computation.")
            # Fall through to computation if deserialization fails
    else:
        print("Cache miss. Running computation.")
        
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
    
    all_research_results: List[ResearchResult] = []
    for el in elements:
        # Extract text content, clean up whitespace, and remove excessive newlines/tabs
        text = el.get_text(strip=True)
        if text:
            # Further cleanup: replace multiple spaces with a single space
            cleaned_text = re.sub(r'\s+', ' ', text)
            names.append(cleaned_text)
            
    for name in names:
            print(f"Finding research for: {name}")
            result = find_research_papers(name, client)
            all_research_results.append(result)
            
    # Filter results that have a valid summary for embedding
    clusterable_results = []
    original_indices = [] # Tracks the index in all_research_results
    
    for i, res in enumerate(all_research_results):
        # Check for valid summary text (not a failure message)
        if res.research_summary and res.cluster_id is None:
            clusterable_results.append(res)
            original_indices.append(i)
    if len(clusterable_results) >= N_CLUSTERS:
        print(f"Generating embeddings for {len(clusterable_results)} items...")
        
        embeddings_list = []
        final_indices = [] # Map back from the embedding list to the original results list
        
        for i, res in zip(original_indices, clusterable_results):
            embedding = generate_embedding(res.research_summary, client)
            if embedding:
                embeddings_list.append(embedding)
                final_indices.append(i) # Store the index of the successful embedding
            else:
                all_research_results[i].cluster_id = -1 # Indicate failure to embed
        if embeddings_list:
            X = np.array(embeddings_list)
            
            # Ensure number of clusters is not greater than the number of samples
            k = min(N_CLUSTERS, X.shape[0])
            
            # --- A. DIMENSIONALITY REDUCTION (PCA) ---
            print(f"Performing PCA to reduce to 2D coordinates...")
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X) # X_2d has shape (N, 2)
            
            # --- B. CLUSTERING (K-Means for Color Coding) ---
            print(f"Performing K-Means clustering with k={k} for coloring...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X) # Clustering still performed on high-dimensional data for better results
            
            # --- C. ASSIGN RESULTS ---
            # Assign cluster labels and 2D coordinates back to the original results
            for idx, cluster_label in enumerate(kmeans.labels_):
                original_index = final_indices[idx]
                
                # Assign cluster ID (for color)
                all_research_results[original_index].cluster_id = int(cluster_label)
                
                # Assign 2D coordinates (for map position)
                all_research_results[original_index].x_coord = float(X_2d[idx, 0])
                all_research_results[original_index].y_coord = float(X_2d[idx, 1])
        
    else:
        print(f"Skipping clustering: Only {len(clusterable_results)} items available. Need at least {N_CLUSTERS}.")
        
    try:
        # Serialize the list of ResearchResult objects to JSON
        results_to_cache = [res.model_dump() for res in all_research_results]
        results_json = json.dumps(results_to_cache)
        r.set(cache_key, results_json, ex=CACHE_EXPIRATION_SECONDS)
        print(f"Stored results in cache key: {cache_key}")
    except Exception as e:
        print(f"Error storing results in cache: {e}")
    return all_research_results

    # except requests.exceptions.RequestException as e:
    #     # Handle network-related errors (DNS failure, connection refused, timeout)
    #     raise HTTPException(status_code=502, detail=f"Failed to fetch external URL: {e}")
    # except Exception as e:
    #     # Handle all other parsing or unexpected errors
    #     raise HTTPException(status_code=500, detail=f"An internal error occurred during scraping: {e}")

@app.get("/health")
def health_check():
    return {'status': 'healthy'}