import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


END_LINGUI = "_stats.txt"
END_ACOUST = "_analysis_DeepSpeech.txt"
MODEL_STRING = "DeepSpeech"



def find_metrics_files(directory):
    lingui_file_path = None  
    acoust_file_path = None  
    wer_file_path = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if END_LINGUI in file:
                lingui_file_path = os.path.join(root, file)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if END_ACOUST in file:
                acoust_file_path = os.path.join(root, file)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("wer_") and file.endswith(".csv") and not "data" in file:
                wer_file_path = os.path.join(root, file)
    
    return lingui_file_path, acoust_file_path, wer_file_path

def parse_txt_to_df(file_path, metric_type):
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                
                # Attempt to convert the value to a float, handle cases where it's not possible
                try:
                    value = float(''.join(filter(lambda x: x.isdigit() or x == '.', value_str)))
                except ValueError:
                    value = value_str
                
                data[key] = value

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Convert all columns except 'Type' to numeric, coercing errors
    df = df.apply(pd.to_numeric, errors='coerce')

    # Add a column to indicate the type of metrics (linguistic or acoustic)
    df['Type'] = metric_type
    return df

def calculate_percentages(df):
    numeric_df = df.select_dtypes(include=['number'])
    aggregated_data = numeric_df.sum()
    percentage_df = numeric_df.divide(aggregated_data) * 100
    return percentage_df, aggregated_data


def plot_individual_datasets(percentage_df, dataset_names, output_folder, prefix, colors):
    for i, dataset_name in enumerate(dataset_names):
        dataset_percentage = percentage_df.iloc[i]
        
        # Ensure the number of colors matches the number of metrics
        if len(colors) < len(dataset_percentage):
            raise ValueError("Not enough colors specified for the number of metrics. Add more colors to the list.")
        
        # Plot the data with the specified colors for each metric
        ax = dataset_percentage.plot(kind='bar', figsize=(10, 6), width=0.8, color=colors[:len(dataset_percentage)])
        ax.set_title(f'{prefix} Metrics - {dataset_name} for {MODEL_STRING} model')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Percentage')
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Create directory for saving the plot
        name_file = f'{prefix.lower()}_metrics_{dataset_name}_{MODEL_STRING}.png'
        plot_path = os.path.join(output_folder, dataset_name)
        os.makedirs(plot_path, exist_ok=True)
        
        full_path = os.path.join(plot_path, name_file)
        
        # Save the plot
        plt.savefig(full_path)
        plt.close()




def main():
    base_directory = os.getcwd()
    output_folder = os.path.join(base_directory, 'comparative_analysis')

    os.makedirs(output_folder, exist_ok=True)

    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]

    all_acoust_dfs = []
    all_ling_dfs = []
    all_wer_dfs = []
    dataset_names = []
    
    for directory in tqdm(subdirectories, desc="Processing Directories", unit="dir"):
        if "test" in directory or 'comparative_analysis' in directory:
            continue

        dataset_name = os.path.basename(directory)
        dataset_names.append(dataset_name)

        lingui_file_path, acoust_file_path, wer_file_path = find_metrics_files(directory)

        if wer_file_path:
            wer_df = pd.read_csv(wer_file_path)
            all_wer_dfs.append(wer_df)

        if lingui_file_path:
            ling_df = parse_txt_to_df(lingui_file_path, "Linguistic")
            all_ling_dfs.append(ling_df)

        if acoust_file_path:
            acoust_df = parse_txt_to_df(acoust_file_path, "Acoustic")
            all_acoust_dfs.append(acoust_df)


    columns_to_drop = [
        'Max duration',
        'Total pauses'
        'Min duration',
        'Sample rate standard deviation', 
        'Pause duration standard deviation', 
        'Pitch standard deviation', 
        'Max pitch', 
        'Min pitch',
        'Total PoS',
        'Noun Count',
        'Verb Count',
        'Adjective Count',
        'Adverb Count',
        'Total entries',
        # 'Total words',
        # 'Unique words',
        # 'Total Sentences',
        # 'Mean Words per Sentence',
    ]

    columns_to_change = {
    'Mean Words per Sentence': 'Avg Words per Sentence',
    
    'Root Type-Token Ratio (RTTR)': 'RTTR',
    'Measure of Textual Lexical Diversity (MTLD)': 'MTLD',
    'Automated Readability Index (ARI)': 'ARI',
    'Average Parse Tree Depth': 'Avg Parse Tree Depth',
    'Average Clauses per Sentence': 'Avg Clauses per Sentence',
    'Average T-Unit Length': 'Avg T-Unit Length',
    
    'Mean duration': 'Avg duration',
    'Mean sample rate': 'Sample rate',
    'Volume standard deviation': 'Volume std',
    'SNR standard deviation': 'SNR std',
    'Mean pause duration': 'Avg pause duration',
    
    'Average WER': 'WER',
    'Total Insertions': 'Insertions',
    'Total Deletions': 'Deletions',
    'Total Substitutions': 'Substitutions',

    'Total Words in Reference': 'Words in Reference',
    'Total Words in Hypothesis': 'Words in Hypothesis',
    'Total Incorrect Words': 'Incorrect Words',
    }

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173'
    ]

    desired_order = [
        'Total words', 'Unique words', 'Total Sentences', 'Avg Words per Sentence',
        'Vocabulary Density', 'Avg Parse Tree Depth', 'Avg Clauses per Sentence',
        'Avg T-Unit Length', 'RTTR', 'MTLD', 'Flesch Reading Ease', 
        'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'ARI', 
        'Coleman-Liau Index'
    ]

    # WER
    combined_wer_df = pd.concat(all_wer_dfs, ignore_index=True)
    combined_wer_df = combined_wer_df.rename(columns=columns_to_change)
    percentage_wer_dfs, _ = calculate_percentages(combined_wer_df)
    
    # Plot individual WER dataset percentages
    plot_individual_datasets(percentage_wer_dfs, dataset_names, output_folder, 'WER', colors)
    
    ax_wer = percentage_wer_dfs.plot(kind='bar', figsize=(15, 8), width=0.8)
    ax_wer.set_title('Percentages of WER and Error Metrics Across Datasets for {MODEL_STRING} model')
    ax_wer.set_xlabel('Dataset Name')
    ax_wer.set_ylabel('Percentage')
    ax_wer.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax_wer.set_xlim(-0.5, len(dataset_names) - 0.5)
    ax_wer.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_wer = os.path.join(output_folder, f'wer_metrics_{MODEL_STRING}.png')
    plt.savefig(plot_path_wer, bbox_inches='tight')
    plt.close()



    # Linguistics
    combined_ling_df = pd.concat(all_ling_dfs, ignore_index=True)
    combined_ling_df = combined_ling_df.drop(columns=columns_to_drop, errors='ignore')
    combined_ling_df = combined_ling_df.rename(columns=columns_to_change)
    combined_ling_df = combined_ling_df[desired_order]
    percentage_ling_dfs, _ = calculate_percentages(combined_ling_df)

    # Plot individual Linguistic dataset percentages
    plot_individual_datasets(percentage_ling_dfs, dataset_names, output_folder, 'Linguistic', colors)

    # General Linguistic plot
    ax_ling = percentage_ling_dfs.plot(kind='bar', figsize=(15, 8), width=0.8, color=colors[:len(percentage_ling_dfs.columns)])
    ax_ling.set_title('Percentages of Linguistic Metrics Across Datasets for {MODEL_STRING} model')
    ax_ling.set_xlabel('Dataset Name')
    ax_ling.set_ylabel('Percentage')
    ax_ling.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax_ling.set_xlim(-0.5, len(dataset_names) - 0.5)
    ax_ling.legend(percentage_ling_dfs.columns, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_ling = os.path.join(output_folder, f'linguistic_metrics_{MODEL_STRING}.png')
    plt.savefig(plot_path_ling, bbox_inches='tight')
    plt.close()



    # Acoustics
    combined_acoust_df = pd.concat(all_acoust_dfs, ignore_index=True)   
    combined_acoust_df = combined_acoust_df.drop(columns=columns_to_drop, errors='ignore')
    combined_acoust_df = combined_acoust_df.rename(columns=columns_to_change)
    percentage_acoust_dfs, _ = calculate_percentages(combined_acoust_df)

    # Plot individual Acoustic dataset percentages
    plot_individual_datasets(percentage_acoust_dfs, dataset_names, output_folder, 'Acoustic', colors)

    ax_acoust = percentage_acoust_dfs.plot(kind='bar', figsize=(15, 8), width=0.8, color=colors[:len(percentage_ling_dfs.columns)])
    ax_acoust.set_title('Percentages of Acoustic Metrics Across Datasets for {MODEL_STRING} model')
    ax_acoust.set_xlabel('Dataset Name')
    ax_acoust.set_ylabel('Percentage')
    ax_acoust.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax_acoust.set_xlim(-0.5, len(dataset_names) - 0.5)
    ax_acoust.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_acoust = os.path.join(output_folder, f'acoustic_metrics_{MODEL_STRING}.png')
    plt.savefig(plot_path_acoust, bbox_inches='tight')
    plt.close()



    print(f"WER plot saved to {plot_path_wer}")
    print(f"Linguistic metrics plot saved to {plot_path_ling}")
    print(f"Acoustic metrics plot saved to {plot_path_acoust}")

if __name__ == "__main__":
    main()
