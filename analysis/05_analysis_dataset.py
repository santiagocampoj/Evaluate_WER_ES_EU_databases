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
        # 'Sample rate standard deviation', 
        # 'Pause duration standard deviation', 
        # 'Pitch standard deviation', 
        'Max pitch', 
        'Min pitch',
        'Total PoS',
        # 'Noun Count',
        # 'Verb Count',
        # 'Adjective Count',
        # 'Adverb Count',
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
    
    'Mean duration': 'Duration avg',
    'Mean sample rate': 'Sample rate',
    'Volume standard deviation': 'Volume std',
    'SNR standard deviation': 'SNR std',
    'Mean pause duration': 'Pause duration avg',
    'Pause duration standard deviation': 'Pause duration std',
    
    'Average WER': 'WER',
    'Total Insertions': 'Insertions',
    'Total Deletions': 'Deletions',
    'Total Substitutions': 'Substitutions',

    'Total Words in Reference': 'Words in Reference',
    'Total Words in Hypothesis': 'Words in Hypothesis',
    # 'Total Incorrect Words': 'Incorrect Words',

    'Adjective Count': 'Adj Count',
    'Adverb Count': 'Adv Count',

    'Duration standar deviation': 'Duration std',
    'Sample rate standard deviation': 'Sample rate std',

    'Pitch standard deviation': 'Pitch std',
    'Mean pauses per audio file': 'Avg pauses per audio file'
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

    lexical_columns = [
       'Total words', 'Unique words',
       'Avg Words per Sentence', 'Vocabulary Density',
       'RTTR', 'MTLD', 'Noun Count', 'Verb Count',
       'Adj Count', 'Adv Count', 'Total PoS', 'Dataset'
    ]

    columns_percentage = {
        'Insertion Percentage': 'Insertion',
        'Deletion Percentage': 'Deletion',
        'Substitution Percentage': 'Substitution',
    }

    # WER
    # combined_wer_df = pd.concat(all_wer_dfs, ignore_index=True)
    # combined_wer_df = combined_wer_df.rename(columns=columns_to_change)

    # combined_wer_df['Insertion Percentage'] = (combined_wer_df['Insertions'] / combined_wer_df['Words in Reference']) * 100
    # combined_wer_df['Deletion Percentage'] = (combined_wer_df['Deletions'] / combined_wer_df['Words in Reference']) * 100
    # combined_wer_df['Substitution Percentage'] = (combined_wer_df['Substitutions'] / combined_wer_df['Words in Reference']) * 100

    # combined_wer_df = combined_wer_df.drop(columns=['Insertions', 'Deletions', 'Substitutions', 'Words in Reference', 'Words in Hypothesis', 'WER', 'Total Incorrect Words'], errors='ignore')
    # combined_wer_df = combined_wer_df.rename(columns=columns_percentage)

    # combined_wer_df['Dataset'] = pd.Categorical(
    #     combined_wer_df['Dataset'], 
    #     categories=desire_dataset_name, 
    #     ordered=True
    # )
    # combined_wer_df = combined_wer_df.sort_values('Dataset')
    
    # ax_wer = combined_wer_df.plot(
    #     x='Dataset', 
    #     kind='bar', 
    #     figsize=(15, 8), 
    #     width=0.8,
    #     color=colors[:len(combined_wer_df.columns) - 1]  # Ensure that colors match the number of columns
    # )
    # ax_wer.set_title(f'Percentages of WER and Error Metrics Across Datasets for ES {MODEL_STRING} model')
    # ax_wer.set_xlabel('Dataset Name')
    # ax_wer.set_ylabel('Percentage')
    # ax_wer.set_xticklabels(combined_wer_df['Dataset'], rotation=45, ha="right")
    
    # ax_wer.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    # plt.tight_layout()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plot_path_wer = os.path.join(output_folder, f'wer_metrics_es_{MODEL_STRING}.png')
    # plt.savefig(plot_path_wer, bbox_inches='tight')




    # # Linguistics
    # combined_ling_df = pd.concat(all_ling_dfs, ignore_index=True)
    # combined_ling_df = combined_ling_df.rename(columns=columns_to_change)

    # # lexical analysis
    # lexical_columns = [
    #    'Dataset','Total words', 'Unique words',
    #    'Avg Words per Sentence', 'Vocabulary Density',
    #    'RTTR', 'MTLD', 'Noun Count', 'Verb Count',
    #    'Adj Count', 'Adv Count', 'Avg Parse Tree Depth', 'Avg Clauses per Sentence'
    # ]

    # lexical_to_remove = [
    #    'Total words', 'Unique words',
    #    'RTTR', 'MTLD', 'Noun Count', 'Verb Count',
    #    'Adj Count', 'Adv Count'
    # ]


    # lexical_df = combined_ling_df[lexical_columns]
    # lexical_df['Unique words'] = (lexical_df['Unique words'] / lexical_df['Total words']) * 100
    # lexical_df['Noun'] = (lexical_df['Noun Count'] / lexical_df['Total words']) * 100
    # lexical_df['Verb'] = (lexical_df['Verb Count'] / lexical_df['Total words']) * 100
    # lexical_df['Adv'] = (lexical_df['Adv Count'] / lexical_df['Total words']) * 100
    # lexical_df['Adj'] = (lexical_df['Adj Count'] / lexical_df['Total words']) * 100
    # lexical_df['Vocabulary Density'] = lexical_df['Vocabulary Density'] * 100
    # lexical_df['Avg Clauses per Sentence'] = lexical_df['Avg Clauses per Sentence'] * 100

    # lexical_df['Dataset'] = pd.Categorical(
    #     lexical_df['Dataset'], 
    #     categories=desire_dataset_name, 
    #     ordered=True
    # )
    
    # lexical_df = lexical_df.sort_values('Dataset')
    # lexical_2 = lexical_df[['RTTR', 'MTLD', 'Dataset']]
    # lexical_df = lexical_df.drop(columns=lexical_to_remove)

    # ax_lexical = lexical_df.plot(
    #     x='Dataset', 
    #     kind='bar', 
    #     figsize=(15, 8), 
    #     width=0.8,
    #     color=colors[:len(lexical_df.columns) - 1]  # Ensure that colors match the number of columns
    # )
    # ax_lexical.set_title(f'Percentages of Lexical and Syntactical Metrics Across Datasets for ES {MODEL_STRING} model')
    # ax_lexical.set_xlabel('Dataset Name')
    # ax_lexical.set_ylabel('Percentage')
    # ax_lexical.set_xticklabels(lexical_df['Dataset'], rotation=45, ha="right")
    
    # ax_lexical.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    # plt.tight_layout()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plot_path_lexical = os.path.join(output_folder, f'lexical_metrics_es_{MODEL_STRING}.png')
    # plt.savefig(plot_path_lexical, bbox_inches='tight')
    


    # # plotting LEXICAL RTTR AND MTLD
    # plt.figure(figsize=(10, 6))
    # for dataset in lexical_2['Dataset']:
    #     subset = lexical_2[lexical_2['Dataset'] == dataset]
    #     plt.plot(subset['RTTR'], subset['MTLD'], marker='o', label=dataset)

    # plt.title('RTTR vs MTLD for Different Datasets')
    # plt.xlabel('RTTR')
    # plt.ylabel('MTLD')
    # plt.legend()
    # plt.grid(True)

    # plot_path_lexical_2 = os.path.join(output_folder, f'RTTR_metrics_es_{MODEL_STRING}.png')
    # plt.savefig(plot_path_lexical_2, bbox_inches='tight')

    # # Plotting the bar plot
    # plt.figure(figsize=(12, 6))
    # bar_width = 0.35
    # index = range(len(lexical_2))
    # plt.bar(index, lexical_2['RTTR'], bar_width, label='RTTR')
    # plt.bar([i + bar_width for i in index], lexical_2['MTLD'], bar_width, label='MTLD')

    # plt.xlabel('Dataset')
    # plt.ylabel('Values')
    # plt.title('RTTR and MTLD for Different Datasets')
    # plt.xticks([i + bar_width / 2 for i in index], lexical_2['Dataset'], rotation=45, ha='right')
    # plt.legend()
    # plt.tight_layout()

    # plot_path_lexical_2_2 = os.path.join(output_folder, f'RTTR_2_metrics_es_{MODEL_STRING}.png')
    # plt.savefig(plot_path_lexical_2_2, bbox_inches='tight')



    # # readability
    # read_columns = [
    #    'Dataset', 'Flesch Reading Ease', 'Flesch-Kincaid Grade Level',
    #    'Gunning Fog Index', 'ARI', 'Coleman-Liau Index'
    # ]
    
    # read_df = combined_ling_df[read_columns]
    # max_value = read_df['Coleman-Liau Index'].max()
    # target_max = 11
    # read_df['Coleman-Liau Index'] = read_df['Coleman-Liau Index'].apply(lambda x: (x / max_value) * target_max)

    # read_df['Dataset'] = pd.Categorical(
    #     read_df['Dataset'], 
    #     categories=desire_dataset_name, 
    #     ordered=True
    # )
    
    # read_df = read_df.sort_values('Dataset')
    # ax_read = read_df.plot(
    #     x='Dataset', 
    #     kind='bar', 
    #     figsize=(15, 8), 
    #     width=0.8,
    #     color=colors[:len(read_df.columns) - 1]  # Ensure that colors match the number of columns
    # )
    # ax_read.set_title(f'Percentages of Lexical and Syntactical Metrics Across Datasets for ES {MODEL_STRING} model')
    # ax_read.set_xlabel('Dataset Name')
    # ax_read.set_ylabel('Percentage')
    # ax_read.set_xticklabels(read_df['Dataset'], rotation=45, ha="right")
    # ax_read.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    # plt.tight_layout()
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # plot_path_lexical = os.path.join(output_folder, f'read_metrics_es_{MODEL_STRING}.png')
    # plt.savefig(plot_path_lexical, bbox_inches='tight')



    # ACOUSTIC
    combined_acoust_df = pd.concat(all_acoust_dfs, ignore_index=True)
    combined_acoust_df = combined_acoust_df.drop(columns=columns_to_drop, errors='ignore')
    combined_acoust_df = combined_acoust_df.drop(columns='Min duration', errors='ignore')
    combined_acoust_df = combined_acoust_df.rename(columns=columns_to_change)
    # print(combined_acoust_df)
    # exit()
    percentage_acoust_dfs, _ = calculate_percentages(combined_acoust_df)

    percentage_acoust_dfs['Dataset'] = pd.Categorical(
        combined_acoust_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    percentage_acoust_dfs = percentage_acoust_dfs.sort_values('Dataset')
    # print(percentage_acoust_dfs.describe())
    percentage_acoust_dfs.to_csv(os.path.join(output_folder, f'acoustic_metrics_es_{MODEL_STRING}.csv'), index=False)
    print(percentage_acoust_dfs)

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
    plot_path_acoust = os.path.join(output_folder, f'acoustic_metrics_es_{MODEL_STRING}.png')
    plt.savefig(plot_path_acoust, bbox_inches='tight')

    

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