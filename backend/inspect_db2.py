import sqlite3
import pandas as pd
conn = sqlite3.connect('researchgapx.db')

print("All Topic Analyses:")
print(pd.read_sql_query("SELECT id, analysis_id, created_at FROM topic_analyses", conn))
print("\nAll Latent Dims:")
print(pd.read_sql_query("SELECT id, analysis_id, dimension_index FROM latent_dimensions", conn).head())
print("\nAll Gap Candidates:")
print(pd.read_sql_query("SELECT analysis_id, COUNT(*) FROM gap_candidates GROUP BY analysis_id", conn))
