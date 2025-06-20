from kaggle_secrets import UserSecretsClient
import openai
import pandas as pd
import json
import re
from prettytable import PrettyTable

# API key
user_secrets = UserSecretsClient()
client = openai.OpenAI(api_key=user_secrets.get_secret("openai"))

# Prompt
prompt = (
    "You are an archaeologist specialized in the Amazon region.\n"
    "List at least 10 known archaeological sites located in the state of Acre, Brazil, "
    "including their approximate latitude and longitude.\n"
    "Format the output as a JSON array with the following fields: name, lat, lon.\n"
    "Example: [{\"name\": \"Site Name\", \"lat\": -X.XXXX, \"lon\": -Y.YYYY}]\n"
    "Focus on geoglyphs and earthworks documented in academic literature or official records.\n"
)

# Call o3
response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": prompt}]
)

# Capture and extract secure JSON from the response
content = response.choices[0].message.content
try:
    sites = json.loads(content)
except:
    match = re.search(r'\[.*\]', content, re.DOTALL)
    sites = json.loads(match.group(0)) if match else []
    
df_benchmark = pd.DataFrame(sites)


# Display the DataFrame in notebook/Kaggle environment
display(df_benchmark)

usage = response.usage
print(f"\nPrompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")