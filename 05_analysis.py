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
                try:
                    value = float(''.join(filter(lambda x: x.isdigit() or x == '.', value_str)))
                except ValueError:
                    print(f"Could not convert {value_str} to float for key {key}. Setting as NaN.")
                    value = None
                data[key] = value

    # Convert to DataFrame
    df = pd.DataFrame([data])
    # print(f"Parsed DataFrame from {file_path}:\n{df}")
    df = df.apply(pd.to_numeric, errors='coerce')
    df['Type'] = metric_type
    return df



def calculate_percentages(df):
    numeric_df = df.select_dtypes(include=['number'])
    aggregated_data = numeric_df.sum()
    percentage_df = numeric_df.divide(aggregated_data) * 100
    return percentage_df, aggregated_data




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
        print(directory)

        dataset_csv = directory.split("/")[-1]
        dataset_name = os.path.basename(directory)
        dataset_names.append(dataset_name)

        lingui_file_path, acoust_file_path, wer_file_path = find_metrics_files(directory)

        if wer_file_path:
            wer_df = pd.read_csv(wer_file_path)
            wer_df['Dataset'] = dataset_csv  
            all_wer_dfs.append(wer_df)

        if lingui_file_path:
            ling_df = parse_txt_to_df(lingui_file_path, "Linguistic")
            ling_df['Dataset'] = dataset_csv  
            all_ling_dfs.append(ling_df)

        if acoust_file_path:
            acoust_df = parse_txt_to_df(acoust_file_path, "Acoustic")
            acoust_df['Dataset'] = dataset_csv  
            all_acoust_dfs.append(acoust_df)
    
    columns_to_drop = [
        'Max duration',
        'Total pauses',
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
        'Unique words',
        'Total Sentences',
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
    # 'Total Incorrect Words': 'Incorrect Words',
    }

    columns_index_dataset = [
        'Europarl-TS',
        'M-AILABS',
        'MintzAI-ST',
        'Common_Voice_v12',
        'ASR-SpCSC',
        'Common_Voice_v9',
        'King-ASR-L-202',
        'OpenSLR',
        'ALBAYZIN2016_ASR',
        'TTS_DB',
        'Parlamento_EJ',
        'Common_Voice_v15',
        '120h_Spanish_Speech',
    ]

    desire_dataset_name =  [
        'Common_Voice_v9',
        'Common_Voice_v12',
        'Common_Voice_v15',
        'M-AILABS',
        '120h_Spanish_Speech',
        'TTS_DB',
        'King-ASR-L-202',
        'MintzAI-ST',
        'Parlamento_EJ',
        'ASR-SpCSC',
        'OpenSLR',
        'ALBAYZIN2016_ASR',
        'Europarl-TS',
    ]

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#7b4173'
    ]

    # desired_order = [
    #     'Total words', 'Noun Count', 'Verb Count', 'Adjective Count', 'Adverb Count', 'Avg Words per Sentence',
    #     'Vocabulary Density', 'RTTR', 'MTLD', 'Avg Parse Tree Depth',
    #     'Avg Clauses per Sentence','Avg T-Unit Length', 'Flesch Reading Ease', 
    #     # 'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'ARI', 
    #     # 'Coleman-Liau Index'
    # ]
    
    desired_order = [
        'Total words', 'Avg Words per Sentence',
        'Vocabulary Density', 'RTTR', 'MTLD', 'Avg Parse Tree Depth',
        'Avg Clauses per Sentence','Avg T-Unit Length', 'Flesch Reading Ease', 
        'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'ARI', 
        'Coleman-Liau Index'
    ]


    # WER
    combined_wer_df = pd.concat(all_wer_dfs, ignore_index=True)
    combined_wer_df = combined_wer_df.rename(columns=columns_to_change)
    percentage_wer_dfs, _ = calculate_percentages(combined_wer_df)
    percentage_wer_dfs['Dataset'] = combined_wer_df['Dataset']

    percentage_wer_dfs['Dataset'] = pd.Categorical(
        percentage_wer_dfs['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    percentage_wer_dfs = percentage_wer_dfs.sort_values('Dataset')
    
    percentage_wer_dfs.to_csv(os.path.join(output_folder, f'wer_metrics_eu_{MODEL_STRING}.csv'), index=False)
    # print(percentage_wer_dfs)

    # PLOT
    ax_wer = percentage_wer_dfs.plot(
        x='Dataset', 
        kind='bar', 
        figsize=(15, 8), 
        width=0.8,
        color=colors[:len(percentage_wer_dfs.columns) - 1]  # Ensure that colors match the number of columns
    )
    ax_wer.set_title(f'Percentages of WER and Error Metrics Across Datasets for EU {MODEL_STRING} model')
    ax_wer.set_xlabel('Dataset Name')
    ax_wer.set_ylabel('Percentage')
    ax_wer.set_xticklabels(percentage_wer_dfs['Dataset'], rotation=45, ha="right")
    
    ax_wer.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_wer = os.path.join(output_folder, f'wer_metrics_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path_wer, bbox_inches='tight')



    # Linguistics
    combined_ling_df = pd.concat(all_ling_dfs, ignore_index=True)
    combined_ling_df = combined_ling_df.drop(columns=columns_to_drop, errors='ignore')
    combined_ling_df = combined_ling_df.rename(columns=columns_to_change)
    combined_ling_df = combined_ling_df[desired_order]
    percentage_ling_dfs, _ = calculate_percentages(combined_ling_df)

    percentage_ling_dfs['Dataset'] = pd.Categorical(
        columns_index_dataset, 
        categories=desire_dataset_name, 
        ordered=True
    )
    percentage_ling_dfs = percentage_ling_dfs.sort_values('Dataset')
    percentage_ling_dfs.to_csv(os.path.join(output_folder, f'linguistic_metrics_eu_{MODEL_STRING}.csv'), index=False)

    # Plot
    if len(colors) < len(percentage_ling_dfs.columns) - 1:
        raise ValueError("Not enough colors specified for the number of metrics. Add more colors to the list.")

    ax_ling = percentage_ling_dfs.plot(
        x='Dataset',
        kind='bar', 
        figsize=(15, 8), 
        width=0.8, 
        color=colors[:len(percentage_ling_dfs.columns) - 1]
    )
    ax_ling.set_title(f'Percentages of Linguistic Metrics Across Datasets for EU {MODEL_STRING} model')
    ax_ling.set_xlabel('Dataset Name')
    ax_ling.set_ylabel('Percentage')
    ax_ling.set_xticklabels(percentage_ling_dfs['Dataset'], rotation=45, ha="right")
    ax_ling.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_ling = os.path.join(output_folder, f'linguistic_metrics_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path_ling, bbox_inches='tight')
    plt.show()




    # Acoustics
    all_acoust_dfs = [df[df['Mean duration'] >= 1] for df in all_acoust_dfs]
    combined_acoust_df = pd.concat(all_acoust_dfs, ignore_index=True)
    combined_acoust_df = combined_acoust_df.drop(columns=columns_to_drop, errors='ignore')
    combined_acoust_df = combined_acoust_df.drop(columns='Min duration', errors='ignore')
    combined_acoust_df = combined_acoust_df.rename(columns=columns_to_change)
    percentage_acoust_dfs, _ = calculate_percentages(combined_acoust_df)

    percentage_acoust_dfs['Dataset'] = pd.Categorical(
        combined_acoust_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    percentage_acoust_dfs = percentage_acoust_dfs.sort_values('Dataset')
    print(percentage_acoust_dfs)
    print(percentage_acoust_dfs.describe())
    # exit()
    percentage_acoust_dfs.to_csv(os.path.join(output_folder, f'acoustic_metrics_eu_{MODEL_STRING}.csv'), index=False)

    # Plot
    ax_acoust = percentage_acoust_dfs.plot(
        x='Dataset', 
        kind='bar', 
        figsize=(15, 8), 
        width=0.9,
        color=colors[:len(percentage_acoust_dfs.columns) - 1]  # Adjust colors to match the number of metrics
    )
    ax_acoust.set_title(f'Percentages of Acoustic Metrics Across Datasets for EU {MODEL_STRING} model')
    ax_acoust.set_xlabel('Dataset Name')
    ax_acoust.set_ylabel('Percentage')
    ax_acoust.set_xticklabels(percentage_acoust_dfs['Dataset'], rotation=45, ha="right")
    ax_acoust.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_acoust = os.path.join(output_folder, f'acoustic_metrics_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path_acoust, bbox_inches='tight')
    plt.show()



    print(f"WER plot saved to {plot_path_wer}")
    print(f"Linguistic metrics plot saved to {plot_path_ling}")
    print(f"Acoustic metrics plot saved to {plot_path_acoust}")

if __name__ == "__main__":
    main()

    # Europarl-TS
    # M-AILABS
    # MintzAI-ST
    # Common_Voice_v12
    # ASR-SpCSC
    # Common_Voice_v9
    # King-ASR-L-202
    # OpenSLR
    # ALBAYZIN2016_ASR
    # TTS_DB
    # Parlamento_EJ
    # Common_Voice_v15
    # 120h_Spanish_Speech