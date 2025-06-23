# OpenAI Agent Integration Example with Function Calling and Structured Output

This document exemplifies how the project flows unification would work using an OpenAI agent with function calling and structured output, to demonstrate mastery of OpenAI API advanced features.

## 1. Authentication and Setup

```python
# Authentication module for direct Python/GEE usage
import ee
import json
from openai import OpenAI
import os
from pathlib import Path

def setup_authentication():
    """Initialize Google Earth Engine and OpenAI clients"""
    # Earth Engine Authentication - Direct service account approach
    service_account_file = os.environ.get("GEE_SERVICE_ACCOUNT_FILE", "gee-service-account.json")
    
    if os.path.exists(service_account_file):
        # Use service account JSON file
        credentials = ee.ServiceAccountCredentials(None, service_account_file)
        ee.Initialize(credentials)
    else:
        # Fallback to user authentication (requires ee.Authenticate() first)
        try:
            ee.Initialize()
        except Exception as e:
            print(f"GEE initialization failed. Run 'earthengine authenticate' first. Error: {e}")
            raise
    
    # OpenAI Authentication
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=openai_key)
    return client
```

## 1.1. Local Setup Requirements

```bash
# Install required packages
pip install earthengine-api openai pandas geopandas

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export GEE_SERVICE_ACCOUNT_FILE="path/to/gee-service-account.json"

# Alternative: use user authentication (interactive)
# earthengine authenticate
```

```python
# Directory structure for local execution
project/
├── main.py                 # Main agent execution
├── gee-service-account.json # GEE service account (gitignore this!)
├── requirements.txt        # Dependencies
└── functions/             # Function implementations
    ├── search.py
    ├── enrich.py
    ├── analyze.py
    └── visualize.py
```

## 2. Enhanced Functions Definition (Tools)

```python
# Enhanced tools definition based on actual project capabilities

tools = [
    {
        "type": "function",
        "name": "search_roi_candidates",
        "description": "Searches for potential archaeological sites based on legends, historical records, and geographical features in any specified region",
        "parameters": {
            "type": "object", 
            "properties": {
                "region": { "type": "string", "description": "Region name or description (e.g., 'Upper Rio Negro', 'Acre, Brazil', 'Central Amazon')" },
                "search_criteria": { "type": "array", "items": {"type": "string"}, "default": ["legends", "historical_records", "topographical_features"], "description": "Search criteria to use" },
                "max_sites": { "type": "number", "default": 5, "description": "Maximum number of sites to return" }
            },
            "required": ["region"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_roi_candidates",
        "description": "Returns candidates within a ROI with remote sensing data enrichment",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": { "type": "number" },
                "longitude": { "type": "number" },
                "radius_m": { "type": "number", "default": 500, "description": "Radius in meters" }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "enrich_with_remote_sensing",
        "description": "Enriches locations with multi-sensor remote sensing data (NDVI, NDWI, NDBI, SRTM, Sentinel-1, MapBiomas, GEDI)",
        "parameters": {
            "type": "object",
            "properties": {
                "sites": { "type": "array", "items": {"type": "object"} },
                "sensors": { "type": "array", "items": {"type": "string"}, "default": ["NDVI", "NDWI", "NDBI", "SRTM", "Sentinel1", "MapBiomas", "GEDI"] }
            },
            "required": ["sites"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "get_benchmark_sites",
        "description": "Gets known archaeological benchmark sites from any specified region for comparison",
        "parameters": {
            "type": "object",
            "properties": {
                "region": { "type": "string", "default": "Global", "description": "Region to get benchmarks from (e.g., 'Acre, Brazil', 'Cusco, Peru', 'Global')" },
                "site_types": { "type": "array", "items": {"type": "string"}, "default": ["geoglyphs", "earthworks", "settlements"], "description": "Types of archaeological sites" }
            },
            "required": [],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "analyze_archaeological_potential",
        "description": "Analyzes candidates against benchmarks using AI expertise in global archaeology and regional patterns",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": { "type": "array", "items": {"type": "object"} },
                "benchmarks": { "type": "array", "items": {"type": "object"} },
                "region_context": { "type": "string", "description": "Geographical and cultural context of the region" }
            },
            "required": ["candidates", "benchmarks"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "compare_environmental_profiles",
        "description": "Performs statistical comparison using z-scores between candidate and benchmark sites",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates_data": { "type": "object" },
                "benchmark_data": { "type": "object" },
                "sensors": { "type": "array", "items": {"type": "string"} }
            },
            "required": ["candidates_data", "benchmark_data"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "generate_satellite_imagery",
        "description": "Generates multi-spectral satellite views (RGB, NIR, NDVI, NDWI, Sentinel-1) for archaeological sites",
        "parameters": {
            "type": "object",
            "properties": {
                "matches": { "type": "array", "items": {"type": "object"} },
                "buffer_m": { "type": "number", "default": 1000 },
                "year": { "type": "number", "default": 2023 }
            },
            "required": ["matches"],
            "additionalProperties": False
        }
    }
]
```

## 3. Agent Orchestration with Error Handling

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

class ArchaeologicalSite(BaseModel):
    name: str
    lat: float
    lon: float
    rationale: Optional[str] = None
    radius_m: Optional[int] = 500

class AnalysisResult(BaseModel):
    matches: List[ArchaeologicalSite]
    summary: str
    confidence_score: float

def run_archaeological_agent(user_input: str):
    """Enhanced agent with error handling and logging"""
    client = setup_authentication()
    
    try:
        response = client.responses.create(
            model="o3",
            input=[{"role": "user", "content": user_input}],
            tools=tools,
            function_call="auto"
        )
        
        # Log usage for monitoring
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[INFO] Tokens used - Prompt: {getattr(usage, 'prompt_tokens', 'N/A')}, "
                  f"Completion: {getattr(usage, 'completion_tokens', 'N/A')}")
        
        return response.output
        
    except Exception as e:
        print(f"[ERROR] Agent execution failed: {e}")
        return {"error": str(e)}
```

## 4. Real User Prompt Examples

```
User: "I want to analyze the region of interest at lat -23.5489, lon -46.6388 with a 1km radius in São Paulo, Brazil. Look for pre-Columbian settlements."
```

```
User: "Find potential archaeological sites near Machu Picchu in Peru based on Inca road networks and topographical analysis."
```

```
User: "Search for Viking settlements in coastal Norway using historical records and landscape signatures typical of Norse occupation."
```

```
User: "Analyze potential Maya sites in the Petén region of Guatemala using LiDAR signatures and known settlement patterns."
```

## 5. Enhanced Structured Output Example

```json
{
  "search_query": "São Paulo pre-Columbian settlements",
  "roi": { 
    "latitude": -23.5489, 
    "longitude": -46.6388, 
    "radius_m": 1000,
    "region": "São Paulo metropolitan area, Brazil"
  },
  "candidates": [
    { 
      "id": "sp_001", 
      "name": "Tietê River Confluence",
      "lat": -23.5489, 
      "lon": -46.6388,
      "rationale": "Strategic location near water sources, elevated terrain suitable for settlements",
      "confidence": 0.78 
    }
  ],
  "remote_sensing": {
    "datasets_used": [
      "COPERNICUS/S2_SR_HARMONIZED",
      "COPERNICUS/S1_GRD", 
      "USGS/SRTMGL1_003",
      "LARSE/GEDI/GEDI02_A_002_MONTHLY"
    ],
    "parameters": {
      "NDVI": 0.342,
      "NDWI": -0.089,
      "NDBI": 0.156,
      "elevation": 760.2,
      "slope": 3.8,
      "sentinel1_vv": -8.2,
      "canopy_height": 12.4,
      "mapbiomas_class": 24
    }
  },
  "benchmark_comparison": {
    "closest_matches": [
      {
        "benchmark_site": "Sambaqui do Piaçaguera",
        "similarity_score": 0.72,
        "key_similarities": ["elevation", "water_proximity", "slope"]
      }
    ],
    "z_score_profile": {
      "NDVI": -0.45,
      "NDWI": 0.12,
      "elevation": 1.85
    }
  },
  "archaeological_assessment": {
    "summary": "Site shows environmental signatures partially consistent with known pre-Columbian settlement patterns. Urban development has significantly altered the landscape, reducing detection confidence.",
    "field_investigation_priority": "Medium",
    "recommended_methods": ["ground-penetrating radar", "test excavations", "artifact surveys"]
  },
  "imagery": {
    "rgb_url": "https://earthengine.googleapis.com/...",
    "ndvi_url": "https://earthengine.googleapis.com/...",
    "radar_url": "https://earthengine.googleapis.com/..."
  }
}
```

## 6. Expected Flow with Real Implementation

1. **User Query Processing**: Agent interprets natural language requests about archaeological investigation
2. **Site Discovery**: Calls `search_roi_candidates` for legend-based or coordinate-based searches  
3. **Data Enrichment**: Uses `enrich_with_remote_sensing` to gather multi-sensor environmental data
4. **Benchmark Comparison**: Retrieves known sites via `get_benchmark_sites` and compares with `compare_environmental_profiles`
5. **Expert Analysis**: Employs `analyze_archaeological_potential` for AI-powered archaeological interpretation
6. **Visualization**: Generates satellite imagery through `generate_satellite_imagery`
7. **Structured Response**: Returns comprehensive JSON with all analysis results

## 7. Advanced Features

- **Multi-sensor Integration**: Combines optical, radar, and LiDAR data sources
- **Statistical Analysis**: Z-score normalization for cross-sensor comparison  
- **Expert Knowledge**: AI trained on global archaeology and regional patterns
- **Geospatial Processing**: Automatic coordinate system handling and spatial analysis
- **Error Handling**: Robust fallback mechanisms for data unavailability
- **Usage Monitoring**: Token counting and performance tracking

## 8. Benefits Beyond Basic Function Calling

- **Domain Expertise**: Incorporates global archaeological and remote sensing knowledge
- **Real-world Data**: Uses actual satellite datasets and known archaeological sites worldwide
- **Scientific Rigor**: Implements established methodologies for site analysis
- **Cultural Sensitivity**: Respects indigenous knowledge and territorial boundaries globally
- **Reproducible Results**: Structured outputs enable validation and peer review

---

This enhanced example demonstrates sophisticated integration of OpenAI agents with real geospatial workflows, showing mastery of both AI orchestration and domain-specific applications in archaeological remote sensing.
