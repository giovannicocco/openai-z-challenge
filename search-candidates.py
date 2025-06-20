
from kaggle_secrets import UserSecretsClient
import openai
from pydantic import BaseModel
from typing import List
import pandas as pd

# Initialize OpenAI client
user_secrets = UserSecretsClient()
client = openai.OpenAI(api_key=user_secrets.get_secret("openai"))

# Prompt: ask o3 for promising but underexplored locations in Nhamini-wi territories (≤200 chars rationale)
prompt = (
    "You are an Amazon explorer and researcher.\n"
    "Based on historical legends, indigenous oral history, and published expedition records, "
    "suggest up to 5 possible locations (latitude and longitude) within the Nhamini-wi region (Upper Rio Negro, near the Brazil/Colombia/Venezuela border) "
    "that could correspond to the legendary trail or its unexplored sites. "
    "Focus on areas that remain little explored archaeologically, according to the scientific literature. "
    "For each, briefly justify your choice referencing myths, remoteness, or lack of fieldwork. "
    "Return your answer as a JSON list with the fields: name, lat, lon, rationale (≤200 characters), and radius_m (fixed value, e.g., 500). "
    "Example: [{\"name\": \"Suggested Area\", \"lat\": 1.2345, \"lon\": -67.8901, \"rationale\": \"...\", \"radius_m\": 500}, ...]"
)

# Define schema with Pydantic (now includes radius_m)
class Area(BaseModel):
    name: str
    lat: float
    lon: float
    rationale: str
    radius_m: int = 500  # default radius in meters

class SuggestedAreas(BaseModel):
    areas: List[Area]

response = client.responses.parse(
    model="o3",
    input=[{"role": "user", "content": prompt}],
    text_format=SuggestedAreas,
)

areas = response.output_parsed.areas

# Ensure full text is shown in the 'rationale' column
pd.set_option('display.max_colwidth', None)


# Helper functions for bbox and WKT
import math
def get_bbox(lat, lon, radius_m):
    # Approximate 1 degree latitude ~ 111.32 km, longitude varies with latitude
    dlat = (radius_m / 111320)
    dlon = (radius_m / (40075000 * math.cos(math.radians(lat)) / 360))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]

def get_bbox_wkt(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return (
        f"POLYGON(("
        f"{min_lon} {min_lat}, "
        f"{min_lon} {max_lat}, "
        f"{max_lon} {max_lat}, "
        f"{max_lon} {min_lat}, "
        f"{min_lon} {min_lat}"  # close polygon
        f"))"
    )

def get_circle_wkt(lat, lon, radius_m, n_points=36):
    # Approximate circle as polygon
    coords = []
    for i in range(n_points+1):
        angle = 2 * math.pi * i / n_points
        dlat = (radius_m / 111320) * math.sin(angle)
        dlon = (radius_m / (40075000 * math.cos(math.radians(lat)) / 360)) * math.cos(angle)
        coords.append(f"{lon + dlon} {lat + dlat}")
    return f"POLYGON(({', '.join(coords)}))"

# Monta DataFrame com bbox e WKT
df = pd.DataFrame([a.model_dump() for a in areas])
df['bbox'] = df.apply(lambda row: get_bbox(row['lat'], row['lon'], row['radius_m']), axis=1)
df['bbox_wkt'] = df['bbox'].apply(get_bbox_wkt)
df['circle_wkt'] = df.apply(lambda row: get_circle_wkt(row['lat'], row['lon'], row['radius_m']), axis=1)


# Display as DataFrame (organized for notebook/Kaggle, agora inclui bbox e WKT)
display(df[['name', 'lat', 'lon', 'radius_m', 'bbox', 'bbox_wkt', 'circle_wkt', 'rationale']].rename(columns={
    'name': 'Name',
    'lat': 'Latitude',
    'lon': 'Longitude',
    'radius_m': 'Radius (m)',
    'bbox': 'BBox [min_lon, min_lat, max_lon, max_lat]',
    'bbox_wkt': 'BBox WKT',
    'circle_wkt': 'Circle WKT',
    'rationale': 'Rationale (≤200 chars)'
}))

# Print model version used
print(f"\n[INFO] OpenAI model used: o3")

# Print coordinates and radius for reference
print("\nSuggested Coordinates (with radius):")
for area in areas:
    print(f"{area.name}: lat {area.lat}, lon {area.lon}, radius {area.radius_m}m")

# Display usage information safely
usage = response.usage
try:
    print("\nPrompt tokens:", getattr(usage, "prompt_tokens", None))
    print("Completion tokens:", getattr(usage, "completion_tokens", None))
    print("Total tokens:", getattr(usage, "total_tokens", None))
except Exception:
    print("\nUsage info:", usage)