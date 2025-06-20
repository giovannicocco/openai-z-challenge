# --- Authenticate with Earth Engine before running this block ---

import ee
import pandas as pd
import time

# --- Earth Engine functions ---

def get_ndvi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
    ndvi = s2.median().select('NDVI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDVI')
    return ndvi.getInfo() if ndvi is not None else None

def get_ndwi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .map(lambda img: img.normalizedDifference(['B3', 'B8']).rename('NDWI')))
    ndwi = s2.median().select('NDWI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDWI')
    return ndwi.getInfo() if ndwi is not None else None

def get_ndbi(lat, lon, year=2023, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(point)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .map(lambda img: img.normalizedDifference(['B11', 'B8']).rename('NDBI')))
    ndbi = s2.median().select('NDBI').reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=10).get('NDBI')
    return ndbi.getInfo() if ndbi is not None else None

def get_srtm_elevation(lat, lon, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    srtm = ee.Image("USGS/SRTMGL1_003")
    elev = srtm.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=30).get('elevation')
    return elev.getInfo() if elev is not None else None

def get_srtm_slope(lat, lon, buffer_m=50):
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    elev = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(elev)
    slope_val = slope.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=30).get('slope')
    return slope_val.getInfo() if slope_val is not None else None

def get_sentinel1_vv(lat, lon, year=2023, buffer_m=1000):
    """
    Returns the average VV backscatter of Sentinel-1.
    
    buffer_m = 1000 by default (>> 50 m from other sensors)
    ──────────────────────────────────────────────────────────
    Using a larger buffer reduces the speckle noise characteristic
    of radar images when spatially averaging. 
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        roi = point.buffer(buffer_m).bounds()
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .select('VV')
        count = s1.size().getInfo()
        if count == 0:
            return None
        s1_img = s1.median()
        vv_value = s1_img.reduceRegion(ee.Reducer.mean(), point, 30).get('VV')
        return vv_value.getInfo() if vv_value is not None else None
    except Exception as e:
        print(f"Sentinel-1 VV error at ({lat}, {lon}): {e}")
        return None

def get_sentinel1_vh(lat, lon, year=2023, buffer_m=1000):
    """
    Returns the average VH backscatter from Sentinel-1.
    
    The 1000m buffer helps to smooth out radar speckle.
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        roi = point.buffer(buffer_m).bounds()
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(roi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select('VH')
        count = s1.size().getInfo()
        if count == 0:
            return None
        s1_img = s1.median()
        vh_value = s1_img.reduceRegion(ee.Reducer.mean(), point, 30).get('VH')
        return vh_value.getInfo() if vh_value is not None else None
    except Exception as e:
        print(f"Sentinel-1 VH error at ({lat}, {lon}): {e}")
        return None

def get_mapbiomas_class(lat, lon, year=2020):
    try:
        point = ee.Geometry.Point(lon, lat)
        img = ee.Image('projects/mapbiomas-raisg/public/collection3/mapbiomas_raisg_panamazonia_collection3_integration_v2') \
            .select(f'classification_{year}')
        value = img.reduceRegion(ee.Reducer.mode(), point, 30).get(f'classification_{year}')
        return value.getInfo() if value is not None else None
    except Exception as e:
        print(f"MapBiomas error at ({lat}, {lon}): {e}")
        return None


def get_gedi_canopy_height(lat, lon):
    """
    Returns mean GEDI canopy height (rh98) for a point.
    If GEDI is not available, fallback to NASA/JPL/global_forest_canopy_height_2005.
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        gedi = (ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
                .filterBounds(point))
        if gedi.size().getInfo() == 0:
            raise ValueError("No GEDI pulses")
        gedi_img = gedi.select('rh98').median()
        value = gedi_img.reduceRegion(
            ee.Reducer.mean(), point, 25).get('rh98')
        val = value.getInfo() if value is not None else None
        if val is not None and val != 0:
            return val
        # If value is None or 0, fallback
        raise ValueError("GEDI returned None or 0")
    except Exception as e:
        print(f"GEDI error at ({lat}, {lon}): {e}. Trying NASA/JPL/global_forest_canopy_height_2005...")
        try:
            # NASA/JPL/global_forest_canopy_height_2005: altura média do dossel em metros (2005)
            point = ee.Geometry.Point([lon, lat])
            canopy_img = ee.Image('NASA/JPL/global_forest_canopy_height_2005')
            value = canopy_img.reduceRegion(
                ee.Reducer.mean(), point, 1000).get('1')
            val = value.getInfo() if value is not None else None
            return val
        except Exception as e2:
            print(f"Fallback canopy height error at ({lat}, {lon}): {e2}")
            return None

# --- Enrich DataFrame with all sensors ---

def enrich_benchmarks_with_all_sensors(
    df,
    ndvi_year=2023,
    ndwi_year=2023,
    ndbi_year=2023,
    s1_year=2023,
    mapbiomas_year=2020,
    buffer_m=50,
    delay=1
):
    ndvi_list = []
    ndwi_list = []
    ndbi_list = []
    elev_list = []
    slope_list = []
    vv_list = []
    vh_list = []
    landclass_list = []
    canopyheight_list = []

    for idx, row in df.iterrows():
        lat, lon = row['lat'], row['lon']
        print(f"Processing {row.get('name', 'site')} ({lat}, {lon})...")
        ndvi_list.append(get_ndvi(lat, lon, ndvi_year, buffer_m))
        ndwi_list.append(get_ndwi(lat, lon, ndwi_year, buffer_m))
        ndbi_list.append(get_ndbi(lat, lon, ndbi_year, buffer_m))
        elev_list.append(get_srtm_elevation(lat, lon, buffer_m))
        slope_list.append(get_srtm_slope(lat, lon, buffer_m))
        vv_list.append(get_sentinel1_vv(lat, lon, s1_year, buffer_m=1000))
        vh_list.append(get_sentinel1_vh(lat, lon, s1_year, buffer_m=1000))
        landclass_list.append(get_mapbiomas_class(lat, lon, mapbiomas_year))
        canopyheight_list.append(get_gedi_canopy_height(lat, lon))
        time.sleep(delay)  # To avoid quota limits

    df['NDVI'] = ndvi_list
    df['NDWI'] = ndwi_list
    df['NDBI'] = ndbi_list
    df['Elevation'] = elev_list
    df['Slope'] = slope_list
    df['Sentinel1_VV'] = vv_list
    df['Sentinel1_VH'] = vh_list
    df['MapBiomas_Class'] = landclass_list
    df['CanopyHeight'] = canopyheight_list
    return df

# --- Usage example ---
# df_benchmark = pd.read_csv("benchmark_sites_acre.csv")  # or from previous cell
df_benchmark = enrich_benchmarks_with_all_sensors(df_benchmark)

# Substitui infinitos por NaN para evitar warnings do pandas
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
df_benchmark.replace([np.inf, -np.inf], np.nan, inplace=True)
display(df_benchmark)