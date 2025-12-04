import numpy as np
import os
import jiwer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

MODEL_STRING = "DeepSpeech"


def compute_wer_details(reference, hypothesis):
    measures = jiwer.compute_measures(reference, hypothesis)
    return round(measures['wer'] * 100, 2)

def process_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '_merge_transcriptions.csv' in file:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df = df
                df['WER'] = df.apply(lambda row: compute_wer_details(row['transcript'], row['hypothesis']), axis=1)
                df = df[['path', 'transcript', 'hypothesis', 'WER','duration']]
    return df


def save_data_and_plot(df, directory):
    # Create directories
    plot_name = directory.split('/')[-1]
    
    plots_directory = os.path.join(directory, 'WER', 'plots')
    text_directory = os.path.join(directory, 'WER')

    os.makedirs(plots_directory, exist_ok=True)
    os.makedirs(text_directory, exist_ok=True)

    print(df)

    if 'WER' in df.columns:
        # WER values greater than 185 to 155
        df.loc[df['WER'] > 185, 'WER'] = 155
        
        unique_wer_values = df['WER'].unique()
        print(f"Unique WER values in {directory}: {unique_wer_values}")
        
        # Print values greater than 100
        high_wer_values = df[df['WER'] > 100]
        if not high_wer_values.empty:
            print(f"WER values greater than 100 in {directory}:")
            print(high_wer_values[['path', 'transcript', 'WER']])
    else:
        print(f"No WER column found in {directory}")
    
    # if WER values greather than 185, set them to 155

    # Plot the bar plot of WER distribution according to the duration
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='duration', y='WER', bins=30, kde=False)
    
    plt.title(f'Duration vs. WER in {plot_name} dataset | {MODEL_STRING} model')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('WER')
    plt.grid(True)

    plt.xlim(df['duration'].min(), df['duration'].max())  # Stick to the min and max of 'duration'
    plt.ylim(df['WER'].min(), df['WER'].max())  # Stick to the min and max of 'WER'
    
    # Save the plot
    plot_file_path = os.path.join(plots_directory, f'{plot_name}_WER_vs_Duration.png')
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Bar plot saved: {plot_file_path}")
    
    

def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]
    
    for directory in tqdm(subdirectories, desc="Processing Directories", unit="dir"):
        if "comparative_analysis" in directory or "test" in directory:
            continue
        # if "Parlamento_EJ" in directory:
        print(f"Processing directory: {directory}")
        df = process_files(directory)
        print(df)
        # exit()

        # print unique values for the WER column
        if 'WER' in df.columns:
            unique_wer_values = df['WER'].unique()
            print(f"Unique WER values in {directory}: {unique_wer_values}")
        else:
            print(f"No WER column found in {directory}")
        
        # exit()
        if not df.empty:
            save_data_and_plot(df, directory)
        else:
            print(f"No data collected in {directory}. Check your directory path or file formats.")

if __name__ == "__main__":
    main()
