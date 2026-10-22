import sqlite3
conn = sqlite3.connect('researchgapx.db')
print("Analysis:", conn.execute("SELECT id, status FROM analyses WHERE id='dadb8792-1bc2-427a-88e9-519338a3c8aa';").fetchall())
print("Topic Analyses:", conn.execute("SELECT id, analysis_id FROM topic_analyses WHERE analysis_id='dadb8792-1bc2-427a-88e9-519338a3c8aa';").fetchall())
print("Topics:", conn.execute("SELECT COUNT(*) FROM topics WHERE topic_analysis_id IN (SELECT id FROM topic_analyses WHERE analysis_id='dadb8792-1bc2-427a-88e9-519338a3c8aa');").fetchall())
print("Gaps:", conn.execute("SELECT COUNT(*) FROM gap_candidates WHERE analysis_id='dadb8792-1bc2-427a-88e9-519338a3c8aa';").fetchall())
