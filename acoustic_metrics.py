import os
import pandas as pd
from tqdm import tqdm

END_ACOUST = "_audio_analysis_DeepSpeech.csv"

def find_metrics_files(directory):
    acoust_file_path = None  
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if END_ACOUST in file:
                acoust_file_path = os.path.join(root, file)

    return acoust_file_path


def new_metrics(df):
    std_duration = df["duration"].std()
    print(f"Duration standar deviation: {std_duration:.2f}\n")



def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]
    
    for directory in tqdm(subdirectories, desc="Processing Directories", unit="dir"):
        if "comparative_analysis" in directory or "test" in directory:
            continue
        
        acoust_file_path = find_metrics_files(directory)
        if acoust_file_path:
            print(f"Processing file: {acoust_file_path}")
            df = pd.read_csv(acoust_file_path)

            new_metrics(df)

if __name__ == "__main__":
    main()
