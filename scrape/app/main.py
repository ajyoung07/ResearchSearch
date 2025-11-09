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
import asyncio

# NEW: Imports for Clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA # Using PCA for 2D coordinate generation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from urllib.parse import urljoin

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

# 1. Define the input data model for the scrape request
class ScrapeRequest(BaseModel):
    url: str
    selector: str # This now targets the link to the bio page (e.g., a.person__image)

# 2. Define the output data model for the research results
class ResearchResult(BaseModel):
    name: str
    # research_summary now holds the person's extracted research areas for embedding
    research_summary: str
    sources: List[Dict[str, str]] # Holds the bio URL
    # Field to hold the assigned cluster ID (None: Needs clustering, >=0: Clustered, -1: Embedding Failed, -2: API Error, -3: General Failure)
    cluster_id: Optional[int] = None 
    # Field to hold the common co-authors (now always None)
    common_co_authors: Optional[str] = None
    # 2D coordinates for the relationship map visualization (from PCA)
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None

# 3. Initialize the FastAPI app
app = FastAPI(
    title="Specialized Two-Level Scraper and Research Area Clusterer API",
    description="Scrapes bio pages for research areas and clusters them using Gemini embeddings."
)

# 4. Configure CORS middleware
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_bio_page(bio_url: str) -> Optional[Dict[str, str]]:
    """
    Synchronous function to fetch a single bio page and extract name and research areas.
    The research area extraction logic is hardcoded for the user's specified website structure.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(bio_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Extract Name (Assumes H1 or H2 holds the person's name on the bio page)
        name_element = soup.select_one('h1') or soup.select_one('h2')
        name = name_element.get_text(strip=True) if name_element else "Unknown Person"
        
        # 2. Extract Research Areas
        research_areas_text = ""
        
        # Hardcoded Selector: article.research-areas-tags > ul > a
        # We look for the containing article, then the unordered list, then the links
        ul_element = soup.select_one('article.research-areas-tags ul')
        
        if ul_element:
            # Select all anchor tags within that specific unordered list
            areas = [a.get_text(strip=True) for a in ul_element.select('a')]
            # Join areas with a distinct separator for vectorization
            research_areas_text = " | ".join(areas) 
            
        return {
            "name": name,
            "research_areas": research_areas_text,
            "bio_url": bio_url
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching bio page {bio_url}: {e}")
        return None
    except Exception as e:
        print(f"General error processing bio page {bio_url}: {e}")
        return None
            

@app.post("/scrape", response_model=List[ResearchResult])
async def scrape_and_find_research(request_data: ScrapeRequest):
    """
    1. Scrapes the main page for bio links using the selector.
    2. Concurrently scrapes each bio link for Name and Research Areas.
    3. Generates TF-IDF vectors from the research areas, performs PCA, and clusters the results.
    4. Caches and returns the results.
    """
    url = request_data.url
    selector = request_data.selector
    
    if not url or not selector:
        raise HTTPException(status_code=400, detail="URL and selector must be provided.")

    # --- Caching Check ---
    cache_key = f"research_map_areas:{url}:{selector}" 
    
    cached_results_json = r.get(cache_key)
    if cached_results_json:
        print("Cache hit! Returning cached results.")
        try:
            cached_data = json.loads(cached_results_json)
            safe_cached_data = []
            for data in cached_data:
                # Ensure cluster_id is present for Pydantic model validation
                if 'cluster_id' not in data:
                    data['cluster_id'] = None
                safe_cached_data.append(data)
            return [ResearchResult(**data) for data in safe_cached_data]
        except Exception as e:
            print(f"Error deserializing cached data: {e}. Re-running computation.")
    else:
        print("Cache miss. Running computation.")
            
    # --- Start Computation (Cache Miss) ---
    
    try:
        # 1. Level 1 Scraping: Get Bio URLs
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, requests.get, url, {'headers': headers, 'timeout': 15})
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        bio_link_elements = soup.select(selector)

        # Extract bio URLs (cap at 10)
        bio_urls = []
        url_set = set() 
        for el in bio_link_elements:
            url_path = el.get('href')
            if url_path:
                # Use urljoin to correctly handle relative paths
                full_url = urljoin(url, url_path)
                
                if full_url not in url_set:
                    bio_urls.append(full_url)
                    url_set.add(full_url)

        if not bio_urls:
            all_research_results = []
            r.set(cache_key, json.dumps([]), ex=CACHE_EXPIRATION_SECONDS)
            return all_research_results

        # 2. Level 2 Scraping: Get Research Areas concurrently
        print(f"Starting concurrent bio page fetching for {len(bio_urls)} people...")
        
        tasks = [loop.run_in_executor(None, fetch_bio_page, bio_url) for bio_url in bio_urls]
        scraped_data: List[Optional[Dict[str, str]]] = await asyncio.gather(*tasks)
        print("Finished concurrent bio page scraping.")
        
        # 3. Prepare ResearchResult objects
        prepared_results: List[ResearchResult] = []
        for data in scraped_data:
            # ONLY include results where the bio page was fetched and research areas were successfully extracted.
            if data and data['research_areas']:
                prepared_results.append(
                    ResearchResult(
                        name=data['name'],
                        research_summary=data['research_areas'], 
                        sources=[{"uri": data['bio_url'], "title": f"Source: {data['name']}'s Bio Page"}],
                        common_co_authors=None,
                        cluster_id=None
                    )
                )

        all_research_results = prepared_results
        
        # 4. Clustering and PCA Logic
        
        clusterable_results = all_research_results
        original_indices = list(range(len(all_research_results)))
        
        # Collect summaries for vectorization
        summaries = [res.research_summary for res in clusterable_results]
        
        # --- A. TF-IDF VECTORIZATION ---
        if len(summaries) > 0:
            print(f"Generating TF-IDF vectors for {len(summaries)} items...")
            
            # Initialize TF-IDF Vectorizer
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                ngram_range=(1, 2), # Use single words and two-word phrases
                max_df=0.85, 
                min_df=1     
            )
            
            # Transform summaries into a dense matrix X for PCA/KMeans
            X_sparse = vectorizer.fit_transform(summaries)
            X = X_sparse.toarray()
            
            # Check if TF-IDF generated a meaningful feature set
            if X.shape[1] > 0:
                
                # --- B. DIMENSIONALITY REDUCTION (PCA) ---
                print(f"Performing PCA to reduce to 2D coordinates...")
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                
                # --- C. CLUSTERING (K-Means for Color Coding) ---
                if X.shape[0] >= N_CLUSTERS: # Run K-Means only if we have enough points
                    k = N_CLUSTERS
                    print(f"Performing K-Means clustering with k={k} for coloring...")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(X)
                    
                    # --- D. ASSIGN RESULTS (K-Means ran) ---
                    for idx, cluster_label in enumerate(kmeans.labels_):
                        original_index = original_indices[idx]
                        all_research_results[original_index].cluster_id = int(cluster_label)
                        all_research_results[original_index].x_coord = float(X_2d[idx, 0])
                        all_research_results[original_index].y_coord = float(X_2d[idx, 1])
                
                else:
                    # --- D. ASSIGN RESULTS (K-Means skipped, but PCA ran) ---
                    print(f"Skipping K-Means: Only {X.shape[0]} items available. Assigning default cluster 0 for visualization.")
                    # Assign cluster_id = 0 and coordinates for visualization
                    for idx in range(X.shape[0]):
                        original_index = original_indices[idx]
                        all_research_results[original_index].cluster_id = 0 # Default cluster 0
                        all_research_results[original_index].x_coord = float(X_2d[idx, 0])
                        all_research_results[original_index].y_coord = float(X_2d[idx, 1])
            else:
                 print("TF-IDF generated an empty feature matrix. Skipping clustering.")
                 for i in original_indices:
                     all_research_results[i].cluster_id = -1 # Indicate failure to vectorize
            
        else:
            print(f"Skipping clustering: No valid summaries found after scraping.")
            for i in original_indices:
                all_research_results[i].cluster_id = None 

        # --- Caching Storage ---
        try:
            results_to_cache = [res.model_dump() for res in all_research_results]
            results_json = json.dumps(results_to_cache)
            r.set(cache_key, results_json, ex=CACHE_EXPIRATION_SECONDS)
            print(f"Stored results in cache key: {cache_key}")
        except Exception as e:
            print(f"Error storing results in cache: {e}")

        return all_research_results

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch external URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred during processing: {e}")

@app.get("/health")
def health_check():
    return {'status': 'healthy'}