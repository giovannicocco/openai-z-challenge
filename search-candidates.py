
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
    "Return your answer as a JSON list with the fields: name, lat, lon, and a short rationale (≤200 characters). "
    "Example: [{\"name\": \"Suggested Area\", \"lat\": 1.2345, \"lon\": -67.8901, \"rationale\": \"...\"}, ...]"
)

 # Define schema with Pydantic
class Area(BaseModel):
    name: str
    lat: float
    lon: float
    rationale: str

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

# Display as DataFrame (organized for notebook/Kaggle)
display(pd.DataFrame([a.model_dump() for a in areas])[['name', 'lat', 'lon', 'rationale']].rename(columns={
    'name': 'Name',
    'lat': 'Latitude',
    'lon': 'Longitude',
    'rationale': 'Rationale (≤200 chars)'
}))

# Print coordinates for reference
print("\nSuggested Coordinates:")
for area in areas:
    print(f"{area.name}: lat {area.lat}, lon {area.lon}")


# Display usage information safely
usage = response.usage
try:
    print("\nPrompt tokens:", getattr(usage, "prompt_tokens", None))
    print("Completion tokens:", getattr(usage, "completion_tokens", None))
    print("Total tokens:", getattr(usage, "total_tokens", None))
except Exception:
    print("\nUsage info:", usage)