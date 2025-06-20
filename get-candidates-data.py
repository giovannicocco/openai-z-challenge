
# Run sensor enrichment on the new candidate areas
# The column 'CanopyHeight' is used for both GEDI and NASA/JPL fallback, matching the benchmark structure
df_candidates = pd.DataFrame([a.model_dump() for a in areas])
num_areas = len(areas)
df_candidates = enrich_benchmarks_with_all_sensors(df_candidates)

# Check: ensure all candidates are present after enrichment
if len(df_candidates) != num_areas:
    print(f"[ERROR] Expected {num_areas} candidates, but df_candidates has {len(df_candidates)} after enrichment!")
    print("Expected IDs:", [a['name'] if hasattr(a, 'name') else a.get('name') for a in areas])
    print("Present IDs:", df_candidates['name'].tolist() if 'name' in df_candidates.columns else df_candidates.index.tolist())
else:
    print(f"[OK] All {num_areas} candidates present in df_candidates.")

display(df_candidates)

# Example: get coordinates of the first candidate
candidate_lat = df_candidates.iloc[0]['lat']
candidate_lon = df_candidates.iloc[0]['lon']