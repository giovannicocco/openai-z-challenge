# Run sensor enrichment on the new candidate areas
# The column 'CanopyHeight' is used for both GEDI and NASA/JPL fallback, matching the benchmark structure

# Log dataset IDs used for enrichment
DATASET_IDS = [
    'COPERNICUS/S2_SR_HARMONIZED',
    'COPERNICUS/S1_GRD',
    'USGS/SRTMGL1_003',
    'projects/mapbiomas-raisg/public/collection3/mapbiomas_raisg_panamazonia_collection3_integration_v2',
    'LARSE/GEDI/GEDI02_A_002_MONTHLY',
    'NASA/JPL/global_forest_canopy_height_2005'
]

print("[INFO] Datasets used in candidate enrichment:")
for ds in DATASET_IDS:
    print(f"  - {ds}")
print()

df_candidates = pd.DataFrame([a.model_dump() for a in areas])
num_areas = len(areas)
df_candidates = enrich_benchmarks_with_all_sensors(df_candidates)

# Add Sentinel-2 thumbnail download column for each candidate
import ee

# Functions to generate download links for different sensors
def get_rgb_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first()
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'bands': ['B4', 'B3', 'B2'],
            'min': 500, 'max': 2500,
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">RGB</a>'
        else:
            return None
    except Exception:
        return None

def get_ndvi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B8', 'B4']).rename('NDVI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': 0, 'max': 1,
            'palette': ['blue', 'white', 'green'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDVI</a>'
        else:
            return None
    except Exception:
        return None

def get_ndwi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B3', 'B8']).rename('NDWI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': -1, 'max': 1,
            'palette': ['brown', 'beige', 'blue'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDWI</a>'
        else:
            return None
    except Exception:
        return None

def get_ndbi_download_url_html(lat, lon, year="2023", month="05"):
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(ee.Geometry.Point(lon, lat)) \
            .filterDate(f'{year}-{month}-01', f'{year}-{month}-31')
        image = collection.first().normalizedDifference(['B11', 'B8']).rename('NDBI')
        region = ee.Geometry.Point(lon, lat).buffer(500).bounds()
        url = image.getThumbURL({
            'min': -1, 'max': 1,
            'palette': ['white', 'gray', 'black'],
            'dimensions': 512,
            'region': region
        })
        if url:
            return f'<a href="{url}" target="_blank">NDBI</a>'
        else:
            return None
    except Exception:
        return None

def get_s1_vv_download_url_html(lat, lon, year="2023"):
    try:
        point = ee.Geometry.Point(lon, lat).buffer(500)
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(point) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV')
        s1_img = s1.median().clip(point)
        url = s1_img.getThumbURL({
            'region': point, 'dimensions': 512, 'min': -25, 'max': 0,
            'palette': ['black', 'white']})
        if url:
            return f'<a href="{url}" target="_blank">Sentinel-1 VV</a>'
        else:
            return None
    except Exception:
        return None

# Download column with all available links (adds only the sensors that are actually available)
def make_download_links(row):
    links = []
    rgb = get_rgb_download_url_html(row['lat'], row['lon'])
    if rgb:
        links.append(rgb)
    ndvi = get_ndvi_download_url_html(row['lat'], row['lon'])
    if ndvi:
        links.append(ndvi)
    ndwi = get_ndwi_download_url_html(row['lat'], row['lon'])
    if ndwi:
        links.append(ndwi)
    ndbi = get_ndbi_download_url_html(row['lat'], row['lon'])
    if ndbi:
        links.append(ndbi)
    s1vv = get_s1_vv_download_url_html(row['lat'], row['lon'])
    if s1vv:
        links.append(s1vv)
    # Add other sensors here if needed
    return ' | '.join(links)

df_candidates['Download'] = df_candidates.apply(make_download_links, axis=1)

# For notebooks: display HTML correctly
from IPython.display import display, HTML
display(HTML(df_candidates.to_html(escape=False)))

# Check: ensure all candidates are present after enrichment
if len(df_candidates) != num_areas:
    print(f"[ERROR] Expected {num_areas} candidates, but df_candidates has {len(df_candidates)} after enrichment!")
    print("Expected IDs:", [a['name'] if hasattr(a, 'name') else a.get('name') for a in areas])
    print("Present IDs:", df_candidates['name'].tolist() if 'name' in df_candidates.columns else df_candidates.index.tolist())
else:
    pass
