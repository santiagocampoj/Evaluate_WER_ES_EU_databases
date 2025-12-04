import pandas as pd
import matplotlib.pyplot as plt
import os

colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#7b4173'
    ]

output_folder = "/home/aholab/santi/Documents/logs/text_analysis/ES/DeepSpeech/comparative_analysis"

DeepSpeech_df = pd.read_csv("/home/aholab/santi/Documents/logs/text_analysis/ES/DeepSpeech/comparative_analysis/wer_metrics_eu_DeepSpeech.csv")
Whisper_LM_df = pd.read_csv("/home/aholab/santi/Documents/logs/text_analysis/ES/Whisper/LM/comparative_analysis/wer_metrics_es_Whisper LM.csv")
Whisper_no_LM_df = pd.read_csv("/home/aholab/santi/Documents/logs/text_analysis/ES/Whisper/no-LM/comparative_analysis/wer_metrics_es_Whisper no-LM.csv")

# Add a 'Model' column to each DataFrame
DeepSpeech_df['Model'] = 'DeepSpeech'
Whisper_LM_df['Model'] = 'Whisper LM'
Whisper_no_LM_df['Model'] = 'Whisper no-LM'

# Combine the DataFrames
combined_df = pd.concat([DeepSpeech_df, Whisper_LM_df, Whisper_no_LM_df])

# Pivot the DataFrame to get the required format for plotting
pivot_df = combined_df.pivot_table(index='Dataset', columns='Model', values=['Insertions', 'Deletions', 'Substitutions'])

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
pivot_df.plot(kind='bar', ax=ax, color=colors)

# Set the plot title and labels
ax.set_title('Percentages of WER and Error Metrics for Different Models')
ax.set_xlabel('Dataset Name')
ax.set_ylabel('Percentage')

# Customize x-axis labels
ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")

# Add legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Add grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot
plot_path = os.path.join(output_folder, 'wer_metrics_comparison.png')
plt.tight_layout()
plt.savefig(plot_path, bbox_inches='tight')
plt.close()