from kaggle_secrets import UserSecretsClient
import openai
import json
import re
from prettytable import PrettyTable

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

# Call the model
response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": prompt}]
)

# Parse the JSON
content = response.choices[0].message.content
try:
    areas = json.loads(content)
except:
    match = re.search(r'\[.*\]', content, re.DOTALL)
    areas = json.loads(match.group(0)) if match else []

# Display as a pretty table
table = PrettyTable()
table.field_names = ["| Name", "Latitude", "Longitude", "Rationale (≤200 chars) |"]
table.hrules = True

for area in areas:
    table.add_row([
        f"| {area['name']}", 
        area['lat'], 
        f"{area['lon']}", 
        f"{area['rationale']} |"
    ])

print(table)

# Print coordinates for reference
print("\nSuggested Coordinates:")
for area in areas:
    print(f"{area['name']}: lat {area['lat']}, lon {area['lon']}")

usage = response.usage
print(f"\nPrompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")