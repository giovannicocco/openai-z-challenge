# Run the sensor enrichment on the new candidate areas

df_candidates = pd.DataFrame(areas)  # From previous cell's 'areas'
df_candidates = enrich_benchmarks_with_all_sensors(df_candidates)
df_candidates.to_csv("nhaminiwi_candidates_enriched.csv", index=False)
display(df_candidates)

candidate_lat = areas[0]['lat']
candidate_lon = areas[0]['lon']