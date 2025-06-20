import openai

# Automatically generates the summary for all analyzed sensors
def generate_sensor_summary(df, label):
    lines = [f"{label} stats:"]
    for col in df.columns:
        if df[col].dtype != object:
            val = df[col].mean()
            lines.append(f"{col}: {val:.3f}")
    return "\n".join(lines)

summary_bench = generate_sensor_summary(df_benchmark, "Benchmark")
summary_cand = generate_sensor_summary(df_candidates, "Candidates")
summary = f"{summary_bench}\n\n{summary_cand}"

prompt = (
    "You are an expert in Amazonian remote sensing and archaeology.\n"
    "Below are summarized environmental parameters for known archaeological sites (benchmarks) and for new candidate locations along the Nhamini-wi trail.\n"
    "Based on this data, compare the candidates to the benchmarks and assess:\n"
    "- Which, if any, of the candidates most closely match the benchmarks?\n"
    "- Are there anomalies or promising signals in the candidate data that warrant field investigation?\n"
    "- Briefly explain the key differences and what they might mean archaeologically.\n"
    "Be concise and analytical, referencing the key parameters (NDVI, NDWI, NDBI, SRTM, slope, Sentinel-1 radar, land cover, canopy height, etc.).\n"
    "After the analysis, output a **JSON array** delimiter with ~~~json … ~~~ ONLY with the candidates that you judge close matches. \n"
    "(one or many). Each object must have:\n"
    "   • name   – the candidate name\n"
    "   • lat    – latitude (decimal)\n"
    "   • lon    – longitude (decimal)\n"
    "   • reason – ≤120-char rationale\n"
    f"\n{summary}\n"
)

response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": prompt}]
)

responseModel = response.choices[0].message.content
print(responseModel)

import json, re
m = re.search(r'~~~json\s*(\[.*\])\s*~~~', responseModel, re.S)
closest = json.loads(m.group(1)) if m else []
print("\nClosest matches parsed:", closest)

# opcional: salva p/ uso posterior
import pandas as pd
pd.DataFrame(closest).to_csv("closest_matches.csv", index=False)

usage = response.usage
print(f"\nPrompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")