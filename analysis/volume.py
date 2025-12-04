import pandas as pd

df = pd.read_csv("/home/aholab/santi/Documents/logs/text_analysis/ES/DeepSpeech/Parlamento_EJ/acoustics/Parlamento_EJ_audio_analysis_DeepSpeech.csv")
print(df)

# remove -inf values
df_cleaned = df[~df["volume"].isin([float('-inf'), float('inf')])]

print(df_cleaned["volume"].describe())