import pandas as pd

df = pd.read_parquet("data/04_llm_temp/django_2020_2025_FINAL_LABELED_12D_LLM_GROQ_v4.parquet")
print(df.columns)

