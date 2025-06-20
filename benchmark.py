from kaggle_secrets import UserSecretsClient
from openai import OpenAI
from pydantic import BaseModel
import pandas as pd

# Load OpenAI API key from Kaggle secrets or environment
try:
    user_secrets = UserSecretsClient()
    openai_key = user_secrets.get_secret("openai")
except Exception:
    import os
    openai_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key) if openai_key else OpenAI()

# Define Pydantic models for structured output
class BenchmarkSite(BaseModel):
    name: str
    lat: float
    lon: float

class BenchmarkSites(BaseModel):
    sites: list[BenchmarkSite]

# Prompt for OpenAI Structured Output (expects a JSON object with a 'sites' key)
prompt = (
    "You are an archaeologist specialized in the Amazon region.\n"
    "List at least 10 known archaeological sites located in the state of Acre, Brazil, "
    "including their approximate latitude and longitude.\n"
    "Return ONLY a JSON object with a 'sites' key, which is a list of objects with fields: name (string), lat (number), lon (number).\n"
    "Example: {\"sites\": [{\"name\": \"Site Name\", \"lat\": -X.XXXX, \"lon\": -Y.YYYY}]}\n"
    "Focus on geoglyphs and earthworks documented in academic literature or official records.\n"
)

# Call OpenAI API and parse with Pydantic Structured Output
response = client.responses.parse(
    model="o3",
    input=[{"role": "user", "content": prompt}],
    text_format=BenchmarkSites,
)

sites = response.output_parsed.sites
df_benchmark = pd.DataFrame([s.model_dump() for s in sites])

# Display DataFrame in notebook/Kaggle environment, fallback to print
try:
    from IPython.display import display
    display(df_benchmark)
except Exception:
    print(df_benchmark)

# Print model version used
print(f"\n[INFO] OpenAI model used: o3")

# Print token usage if available
usage = getattr(response, "usage", None)
if usage:
    print(f"\nPrompt tokens: {getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', None))}")
    print(f"Completion tokens: {getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', None))}")
    print(f"Total tokens: {getattr(usage, 'total_tokens', None)}")