# Amazon Archaeological Site Detection Project

## Overview
This project uses artificial intelligence (OpenAI's o3 model) combined with remote sensing data from Google Earth Engine to identify potential archaeological sites in the Amazon rainforest, specifically in the Nhamini-wi region of the Upper Rio Negro basin (Brazil/Colombia/Venezuela border).

## Project Structure

### 1. Authentication (`auth.py`)
- Authenticates with Google Earth Engine using service account credentials
- Retrieves API keys from Kaggle Secrets for security
- Sets up the connection to Earth Engine services

### 2. Benchmark Site Generation (`benchmark.py`)
- Uses OpenAI's o3 model to generate a list of known archaeological sites in Acre, Brazil
- Focuses on geoglyphs and earthworks documented in academic literature
- Creates a reference dataset with coordinates (latitude/longitude) for comparison
- Outputs data in both CSV format and formatted tables

### 3. Remote Sensing Data Collection (`get-benchmark-data.py`)
This is the core module that enriches location data with multiple satellite sensors:

#### Environmental Parameters Collected:
- **NDVI (Normalized Difference Vegetation Index)** - Vegetation health indicator
- **NDWI (Normalized Difference Water Index)** - Water content and moisture
- **NDBI (Normalized Difference Built-up Index)** - Built environment detection
- **SRTM Elevation & Slope** - Topographical characteristics
- **Sentinel-1 Radar (VV/VH)** - Penetrates vegetation to detect subsurface features
- **MapBiomas Land Cover Classification** - Land use classification
- **GEDI Canopy Height** - Forest canopy structure from LiDAR

#### Data Sources:
- Sentinel-2 optical imagery (Copernicus program)
- Sentinel-1 synthetic aperture radar
- SRTM digital elevation model
- MapBiomas Pan-Amazon land cover data
- GEDI (Global Ecosystem Dynamics Investigation) LiDAR


### 4. Candidate Site Discovery (`search-candidates.py`)
- Uses AI to suggest promising but underexplored locations in the Nhamini-wi region
- Based on historical legends, indigenous oral history, and expedition records
- Generates hypothetical coordinates for areas that warrant archaeological investigation
- Provides rationale for each suggested location
- **Each candidate footprint includes a center (latitude/longitude) and a fixed radius (e.g., 500m), allowing representation as a circle or bounding box (bbox/WKT) for spatial analysis, as required by the OpenAI to Z Challenge.**

### 5. Candidate Data Processing (`get-candidates-data.py`)
- Applies the same remote sensing analysis to candidate locations
- Creates comparable datasets between known sites and potential discoveries

### 6. Comparative Analysis (`compare.py`)
- Performs statistical comparison between benchmark sites and candidate locations
- Normalizes sensor data using z-scores for fair comparison
- Creates visualization plots showing environmental parameter profiles
- Identifies which candidates most closely match known archaeological sites

### 7. AI-Powered Site Assessment (`analyze-candidates-data.py`)
- Uses OpenAI's o3 model to analyze environmental data patterns
- Compares candidate sites against benchmark archaeological sites
- Provides expert-level interpretation of remote sensing anomalies
- Outputs JSON-formatted results with closest matches and archaeological significance

### 8. Satellite Imagery Visualization (`get-image-for-closest-match.py`)
- Generates multi-spectral satellite views of promising locations
- Creates composite images including:
  - True color RGB
  - Near-infrared false color
  - NDVI vegetation index
  - NDWI water index
  - Sentinel-1 radar backscatter
- Provides visual inspection capabilities for identified sites

## Methodology

### Remote Sensing Approach
The project employs a multi-sensor approach to characterize archaeological sites:
1. **Optical sensors** detect vegetation anomalies and surface features
2. **Radar sensors** penetrate forest canopy to reveal subsurface structures
3. **LiDAR data** provides precise elevation and canopy measurements
4. **Land cover data** identifies human-modified landscapes

### AI Integration
- **Site Discovery**: AI generates hypotheses about unexplored locations based on historical and cultural knowledge
- **Pattern Recognition**: Machine learning identifies environmental signatures of known archaeological sites
- **Expert Analysis**: AI provides archaeological interpretation of remote sensing anomalies

### Statistical Analysis
- Z-score normalization enables comparison across different sensor types
- Pattern matching identifies candidate sites with similar environmental signatures to known archaeological locations

## Applications
- **Archaeological Survey Planning**: Prioritizes areas for field investigation
- **Cultural Heritage Protection**: Identifies sites at risk from deforestation or development
- **Indigenous Territory Mapping**: Documents traditional landscapes and settlement patterns
- **Environmental Archaeology**: Studies human-environment interactions in the Amazon

## Technical Requirements
- Google Earth Engine account with API access
- OpenAI API key (o3 model access)
- Python environment with geospatial libraries
- Kaggle environment for secure credential management

## Data Outputs
- CSV files with enriched location data
- Comparative analysis plots
- Satellite imagery composites
- JSON reports with archaeological assessments
- Prioritized lists of sites for field verification

This project demonstrates the integration of artificial intelligence with remote sensing technology for archaeological discovery in one of the world's most challenging environments - the Amazon rainforest.

## Setting Up Kaggle Secrets

To keep credentials secure when running in a Kaggle notebook, create two secrets:

1. **`openai`** – your OpenAI API key.
2. **`service_account`** – the Google Cloud service account JSON.

Open the **Add-ons → Secrets** dialog in Kaggle and add these entries. The scripts will automatically load them via `UserSecretsClient`.

## Notebook Execution Order

Run the following scripts in this order within your Kaggle notebook:

1. `benchmark.py` – generate reference sites.
2. `auth.py` – authenticate with Earth Engine.
3. `get-benchmark-data.py` – collect remote sensing data for benchmarks.
4. `search-candidates.py` – propose potential locations.
5. `get-candidates-data.py` – gather data for candidates.
6. `compare.py` – statistically compare results.
7. `analyze-candidates-data.py` – use OpenAI to interpret findings.
8. `get-image-for-matches.py` – visualize imagery for top matches.
