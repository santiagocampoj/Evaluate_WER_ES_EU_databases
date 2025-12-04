import os
import pandas as pd
from tqdm import tqdm

def find_metrics_files(directory):
    ref_hyp_file_path = None
    acoust_file_path = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_es_model_references_hypotheses.csv") or file.endswith("_eu_model_references_hypotheses.csv"):
                ref_hyp_file_path = os.path.join(root, file)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_audio_analysis_DeepSpeech.csv"):
                acoust_file_path = os.path.join(root, file)
    
    return ref_hyp_file_path, acoust_file_path

def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]
    
    for directory in tqdm(subdirectories, desc="Processing Directories", unit="dir"):
        if "comparative_analysis" in directory or "test" in directory:
            continue
        print(f"Processing directory: {directory}")
        references_hypotheses, acoustic = find_metrics_files(directory)
        
        # load csv
        references_hypotheses_df = pd.read_csv(references_hypotheses)
        acoustic_df = pd.read_csv(acoustic)

        # output path
        dataset_name = os.path.basename(directory)
        plots_directory = os.path.join(directory, 'linguistics', 'raw_data')
        os.makedirs(plots_directory, exist_ok=True)

        references_hypotheses_df['filename'] = references_hypotheses_df['path'].apply(os.path.basename)
        acoustic_df['filename'] = acoustic_df['file_path'].apply(os.path.basename)

        merged_df = pd.merge(references_hypotheses_df, acoustic_df[['filename', 'duration']], on='filename', how='left')
        merged_df.drop(columns=['filename'], inplace=True)
        merged_df = merged_df.dropna()

        #save
        csv_save_path = os.path.join(plots_directory, f'{dataset_name}_merge_transcriptions.csv')
        merged_df.to_csv(csv_save_path, index=False)
        print(f"Saving CSV to: {csv_save_path}")

if __name__ == "__main__":
    main()
