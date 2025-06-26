# get-image-for-matches.py

import ee

def plot_multiple_satellite_views(lat, lon, buffer_m=1000, year=2023):
    point = ee.Geometry.Point(lon, lat).buffer(buffer_m)
    print("[INFO] Using dataset_id: COPERNICUS/S2_SR_HARMONIZED (Sentinel-2) for RGB, Infrared (NIR), NDVI, NDWI")
    
    # Get Sentinel-2 collection and select least cloudy image
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(point)
           .filterDate(f'{year}-01-01', f'{year}-12-31')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
           .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    img = s2_collection.first().clip(point)
    
    # Get Sentinel-2 scene ID
    try:
        s2_id = img.get('PRODUCT_ID')
        if s2_id is None:
            s2_id = img.get('system:index')
        s2_scene_id = s2_id.getInfo() if s2_id is not None else "N/A"
    except Exception as e:
        print(f"[WARNING] Could not get Sentinel-2 scene ID: {e}")
        s2_scene_id = "N/A"
    
    # URLs for each composite
    urls = {}
    urls['RGB'] = img.select(['B4', 'B3', 'B2']).getThumbURL({
        'region': point, 'dimensions': 512, 'min': 500, 'max': 2500})
    urls['Infrared (NIR)'] = img.select(['B8', 'B4', 'B3']).getThumbURL({
        'region': point, 'dimensions': 512, 'min': 500, 'max': 2500})
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    urls['NDVI'] = ndvi.getThumbURL({
        'region': point, 'dimensions': 512, 'min': 0, 'max': 1,
        'palette': ['blue', 'white', 'green']})
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    urls['NDWI'] = ndwi.getThumbURL({
        'region': point, 'dimensions': 512, 'min': -1, 'max': 1,
        'palette': ['brown', 'beige', 'blue']})
    
    print("[INFO] Using dataset_id: COPERNICUS/S1_GRD (Sentinel-1) for Sentinel-1 VV")
    # Sentinel-1 VV
    s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(point) \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .select('VV')
    
    # Check if Sentinel-1 collection has any images
    s1_size = s1_collection.size()
    try:
        s1_count = s1_size.getInfo()
        if s1_count > 0:
            s1_img = s1_collection.median().clip(point)
            
            # Get Sentinel-1 scene ID (using first image from collection)
            try:
                s1_first = s1_collection.first()
                s1_id = s1_first.get('system:index')
                s1_scene_id = s1_id.getInfo() if s1_id is not None else "N/A"
            except Exception as e:
                print(f"[WARNING] Could not get Sentinel-1 scene ID: {e}")
                s1_scene_id = "N/A"
            
            urls['Sentinel-1 VV'] = s1_img.getThumbURL({
                'region': point, 'dimensions': 512, 'min': -25, 'max': 0,
                'palette': ['black', 'white']})
        else:
            print(f"[WARNING] No Sentinel-1 images found for this location and time period")
            s1_scene_id = "No images available"
            # Skip adding Sentinel-1 VV to urls
    except Exception as e:
        print(f"[WARNING] Error processing Sentinel-1 data: {e}")
        s1_scene_id = "Error processing"
        # Skip adding Sentinel-1 VV to urls
    
    # Plot all images
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    
    # Calculate subplot dimensions based on number of images
    num_images = len(urls)
    if num_images <= 3:
        rows, cols = 1, num_images
        figsize = (4 * num_images, 4)
    else:
        rows, cols = 2, 3
        figsize = (12, 8)
    
    plt.figure(figsize=figsize)
    
    plot_idx = 1
    for name, url in urls.items():
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            im = Image.open(BytesIO(response.content))
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(im)
            plt.title(name)
            plt.axis('off')
            plot_idx += 1
        except Exception as e:
            print(f"[WARNING] Could not load image for {name}: {e}")
            # Continue with other images
    
    plt.tight_layout()
    plt.show()
    
    # Print scene IDs for reference
    print(f"\n[INFO] Scene IDs used:")
    print(f"  Sentinel-2: {s2_scene_id}")
    print(f"  Sentinel-1: {s1_scene_id}")

# Example usage for all matches using the in-memory df_matches DataFrame:
import pandas as pd
# Log dataset ID if available in df_matches
if 'df_matches' in globals() and df_matches is not None and not df_matches.empty:
    dataset_id = None
    # Try to get dataset_id from DataFrame attribute or column
    if hasattr(df_matches, 'dataset_id'):
        dataset_id = getattr(df_matches, 'dataset_id', None)
    elif 'dataset_id' in df_matches.columns:
        dataset_id = df_matches['dataset_id'].iloc[0]
    if dataset_id:
        print(f"[INFO] Using dataset_id: {dataset_id}")
    for _, m in df_matches.iterrows():
        print(f"\n[INFO] Generating images for: {m['name']} (lat: {m['lat']}, lon: {m['lon']})")
        plot_multiple_satellite_views(m['lat'], m['lon'], buffer_m=1000, year=2023)
else:
    print("No match found to generate images.")