from openai import OpenAI
import os
import pandas as pd
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    openai_key = user_secrets.get_secret("openai")
except Exception:
    openai_key = os.environ.get("OPENAI_API_KEY")
from pydantic import BaseModel
try:
    from IPython.display import display
except ImportError:
    display = None

# Generates the mean summary for all benchmark sensors
def generate_sensor_summary(df, label):
    lines = [f"{label} stats (mean):"]
    for col in df.columns:
        if df[col].dtype != object:
            val = df[col].mean()
            lines.append(f"{col}: {val:.3f}")
    return "\n".join(lines)

# Generates a detailed summary for all candidates
def generate_candidates_detail(df):
    lines = ["Candidates:"]
    for idx, row in df.iterrows():
        vals = []
        for col in df.columns:
            if df[col].dtype != object:
                vals.append(f"{col}: {row[col]:.3f}")
            else:
                vals.append(f"{col}: {row[col]}")
        lines.append("- " + ", ".join(vals))
    return "\n".join(lines)

summary_bench = generate_sensor_summary(df_benchmark, "Benchmark")
summary_cand = generate_candidates_detail(df_candidates)
summary = f"{summary_bench}\n\n{summary_cand}"

prompt = (
    "You are an expert in Amazonian remote sensing and archaeology.\n"
    "Below are summarized environmental parameters for known archaeological sites (benchmarks) and for new candidate locations along the Nhamini-wi trail.\n"
    "Based on this data, compare the candidates to the benchmarks and assess:\n"
    "- Which, if any, of the candidates most closely match the benchmarks?\n"
    "- Are there anomalies or promising signals in the candidate data that warrant field investigation?\n"
    "- Briefly explain the key differences and what they might mean archaeologically.\n"
    "Be concise and analytical, referencing the key parameters (NDVI, NDWI, NDBI, SRTM, slope, Sentinel-1 radar, land cover, canopy height, etc.).\n"
    "Return ONLY a JSON object with a 'matches' key, which is a list of the closest candidate(s) to the benchmarks. Each match must have: name, lat, lon, reason. If none, return an empty list.\n"
    f"\n{summary}\n"
)
    # Defines the schema for Structured Outputs
schema = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "lat":    {"type": "number"},
                    "lon":    {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["name", "lat", "lon", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["matches"],
    "additionalProperties": False
}

# --- Structured Output with Pydantic ---
client = OpenAI(api_key=openai_key) if openai_key else OpenAI()

class ClosestMatch(BaseModel):
    name: str
    lat: float
    lon: float
    reason: str

class ClosestMatches(BaseModel):
    matches: list[ClosestMatch]

response = client.responses.parse(
    model="o3",
    input=[{"role": "user", "content": prompt}],
    text_format=ClosestMatches,
)

matches = response.output_parsed.matches

print("\nMatches:")
for m in matches:
    print(f"- {m.name} (lat: {m.lat}, lon: {m.lon})\n  Reason: {m.reason}\n")

# Display the result as a DataFrame in Kaggle/notebook environments (optional)

import pandas as pd

# Ensure full text is shown in the 'reason' column (rationale)
pd.set_option('display.max_colwidth', None)

df_matches = pd.DataFrame([m.model_dump() for m in matches])
try:
    from IPython.display import display
    display(df_matches)
except Exception:
    print(df_matches)

# Display usage information safely (as in search-candidates.py)
usage = getattr(response, "usage", None)
try:
    print("\nPrompt tokens:", getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)))
    print("Completion tokens:", getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)))
    print("Total tokens:", getattr(usage, "total_tokens", None))
except Exception:
    print("\nUsage info:", usage)