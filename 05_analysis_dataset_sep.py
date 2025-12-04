import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        'Noun Count',
        'Verb Count',
        'Adjective Count',
        'Adverb Count',
        'Total entries',
        # 'Total words',
        'Unique words',
        'Total Sentences',
        # 'Mean Words per Sentence',
        'Total Words in Reference',
        'Total Words in Hypothesis',
        'Total Incorrect Words',
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
    'Duration standar deviation': 'Duration std',

    'Mean volume': 'Avg volume',

    'Volume standard deviation': 'Volume std',
    'SNR standard deviation': 'SNR std',

    'Mean pauses per audio file': 'Avg pauses per audio file',
    'Mean pause duration': 'Avg pause duration',
    'Pause duration standard deviation': 'Pause duration std',
    'Pitch standard deviation': 'Pitch std',
    'Mean pitch': 'Avg pitch',
    
    'Average WER': 'WER',
    'Total Insertions': 'Insertions',
    'Total Deletions': 'Deletions',
    'Total Substitutions': 'Substitutions',

    'Total Words in Reference': 'Words in Reference',
    'Total Words in Hypothesis': 'Words in Hypothesis',
    # 'Total Incorrect Words': 'Incorrect Words',
    'Sample rate standard deviation': 'Sample rate std',
    'Std Dev of Words per Sentence': 'Words per Sentence std'
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
    # colors = plt.cm.get_cmap('tab20').colors

    columns_percentage = {
        'Insertion Percentage': 'Insertion',
        'Deletion Percentage': 'Deletion',
        'Substitution Percentage': 'Substitution',
    }


    # WER
    combined_wer_df = pd.concat(all_wer_dfs, ignore_index=True)
    combined_wer_df = combined_wer_df.rename(columns=columns_to_change)

    combined_wer_df['Insertion Percentage'] = (combined_wer_df['Insertions'] / combined_wer_df['Words in Reference']) * 100
    combined_wer_df['Deletion Percentage'] = (combined_wer_df['Deletions'] / combined_wer_df['Words in Reference']) * 100
    combined_wer_df['Substitution Percentage'] = (combined_wer_df['Substitutions'] / combined_wer_df['Words in Reference']) * 100
    
    combined_wer_df = combined_wer_df.drop(columns=['Insertions', 'Deletions', 'Substitutions', 'Words in Reference', 'Words in Hypothesis', 'WER', 'Total Incorrect Words', 'Hallucination Count'], errors='ignore')
    combined_wer_df = combined_wer_df.rename(columns=columns_percentage)

    combined_wer_df['Dataset'] = pd.Categorical(
        combined_wer_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    combined_wer_df = combined_wer_df.sort_values('Dataset')
    
    ax_wer = combined_wer_df.plot(
        x='Dataset', 
        kind='bar', 
        figsize=(15, 8), 
        width=0.8,
        color=colors[:len(combined_wer_df.columns) - 1]  # Ensure that colors match the number of columns
    )
    ax_wer.set_title(f'Percentages of WER and Error Metrics for EU {MODEL_STRING} model')
    ax_wer.set_xlabel('Dataset Name')
    ax_wer.set_ylabel('Percentage')
    ax_wer.set_xticklabels(combined_wer_df['Dataset'], rotation=45, ha="right")
    
    ax_wer.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_wer = os.path.join(output_folder, f'wer_metrics_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path_wer, bbox_inches='tight')
    combined_wer_df.to_csv(os.path.join(output_folder, f'wer_metrics_eu_{MODEL_STRING}.csv'), index=False)



    # # Linguistics
    combined_ling_df = pd.concat(all_ling_dfs, ignore_index=True)
    combined_ling_df = combined_ling_df.rename(columns=columns_to_change)

    # lexical analysis
    lexical_columns = [
       'Dataset','Total words', 'Words per Sentence std',
       'Avg Words per Sentence', 'Vocabulary Density',
       'RTTR', 'MTLD', 'Noun Count', 'Verb Count',
       'Adj Count', 'Adv Count', 'Avg Parse Tree Depth', 'Avg Clauses per Sentence'
    ]

    lexical_to_remove = [
       'Total words',
       'RTTR', 'MTLD', 'Noun Count', 'Verb Count',
       'Adj Count', 'Adv Count', 'Noun', 'Verb', 'Aux', 'Adv', 'Adj'
    ]


    lexical_df = combined_ling_df[lexical_columns]
    lexical_df['Noun'] = (lexical_df['Noun Count'] / lexical_df['Total words']) * 100
    lexical_df['Verb'] = (lexical_df['Verb Count'] / lexical_df['Total words']) * 100
    lexical_df['Aux'] = (lexical_df['Aux Count'] / lexical_df['Total words']) * 100
    lexical_df['Adv'] = (lexical_df['Adv Count'] / lexical_df['Total words']) * 100
    lexical_df['Adj'] = (lexical_df['Adj Count'] / lexical_df['Total words']) * 100
    lexical_df['Avg Clauses per Sentence'] = lexical_df['Avg Clauses per Sentence'] * 100
    lexical_df['Vocabulary Density'] = (lexical_df['Vocabulary Density'] * 100)

    lexical_df['Dataset'] = pd.Categorical(
        lexical_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    
    lexical_df = lexical_df.sort_values('Dataset')
    voca_den_df = lexical_df[['Vocabulary Density', 'Dataset']]
    lexical_2 = lexical_df[['RTTR', 'MTLD', 'Dataset']]
    morfology_voca_df = lexical_df[['Noun', 'Verb', 'Adv', 'Adj', 'Dataset']]

    lexical_df = lexical_df.drop(columns=lexical_to_remove)

    fig, ax_lexical = plt.subplots(figsize=(15, 8))

    ax_lexical.plot(lexical_df['Dataset'], lexical_df['Avg Words per Sentence'], marker='o', color='blue', label='Avg Words per Sentence')

    upper_bound = lexical_df['Avg Words per Sentence'] + lexical_df['Words per Sentence std']
    lower_bound = lexical_df['Avg Words per Sentence'] - lexical_df['Words per Sentence std']

    ax_lexical.fill_between(lexical_df['Dataset'], lower_bound, upper_bound, color='blue', alpha=0.2, label='Words per Sentence Std Dev')

    colors = ['green', 'red', 'purple']
    metrics = ['Vocabulary Density', 'Avg Parse Tree Depth', 'Avg Clauses per Sentence']

    for i, column in enumerate(metrics):
        ax_lexical.plot(lexical_df['Dataset'], lexical_df[column], marker='o', color=colors[i], label=column)

    ax_lexical.set_title('Syntactical Metrics for EU Datasets')
    ax_lexical.set_xlabel('Dataset Name')
    ax_lexical.set_ylabel('Percentage')
    ax_lexical.set_xticklabels(lexical_df['Dataset'], rotation=45, ha="right")
    ax_lexical.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_lexical = os.path.join(output_folder, 'syntactical_metrics_eu.png')
    plt.savefig(plot_path_lexical, bbox_inches='tight')
    plt.close()
    lexical_df.to_csv(os.path.join(output_folder, 'syntactical_metrics_eu.csv'), index=False)

    
    # morphology
    ax_mophology = morfology_voca_df.plot(
        x='Dataset', 
        kind='bar', 
        figsize=(15, 8), 
        width=0.8,
        color=colors[:len(lexical_df.columns) - 1]  # Ensure that colors match the number of columns
    )
    ax_mophology.set_title(f'Count of Morphologycal Metrics EU Datasets')
    ax_mophology.set_xlabel('Dataset Name')
    ax_mophology.set_ylabel('Count')
    ax_mophology.set_xticklabels(lexical_df['Dataset'], rotation=45, ha="right")
    
    ax_mophology.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_morphology = os.path.join(output_folder, f'morphology_metrics_eu.png')
    plt.savefig(plot_path_morphology, bbox_inches='tight')
    morfology_voca_df.to_csv(os.path.join(output_folder, f'morphology_metrics_eu.csv'), index=False)
    


    # # plotting LEXICAL RTTR AND MTLD
    plt.figure(figsize=(10, 6))
    for dataset in lexical_2['Dataset']:
        subset = lexical_2[lexical_2['Dataset'] == dataset]
        plt.plot(subset['RTTR'], subset['MTLD'], marker='o', label=dataset)

    plt.title('RTTR vs MTLD for EU Datasets')
    plt.xlabel('RTTR')
    plt.ylabel('MTLD')
    plt.legend()
    plt.grid(True)

    plot_path_lexical_2 = os.path.join(output_folder, f'RTTR_metrics_eu.png')
    plt.savefig(plot_path_lexical_2, bbox_inches='tight')

    # Plotting the bar plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(lexical_2))
    plt.bar(index, lexical_2['RTTR'], bar_width, label='RTTR')
    plt.bar([i + bar_width for i in index], lexical_2['MTLD'], bar_width, label='MTLD')

    plt.xlabel('Dataset')
    plt.ylabel('Values')
    plt.title('RTTR and MTLD for Different Datasets')
    plt.xticks([i + bar_width / 2 for i in index], lexical_2['Dataset'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    plot_path_lexical_2_2 = os.path.join(output_folder, f'RTTR_2_metrics_eu.png')
    plt.savefig(plot_path_lexical_2_2, bbox_inches='tight')
    lexical_2.to_csv(os.path.join(output_folder, f'rttr_mtld_metrics_eu.csv'), index=False)



    # # readability
    read_columns = [
       'Dataset', 'Flesch Reading Ease', 'Flesch-Kincaid Grade Level',
       'Gunning Fog Index', 'ARI', 'Coleman-Liau Index'
    ]
    
    read_df = combined_ling_df[read_columns]
    max_value = read_df['Coleman-Liau Index'].max()
    target_max = 11
    read_df['Coleman-Liau Index'] = read_df['Coleman-Liau Index'].apply(lambda x: (x / max_value) * target_max)

    read_df['Dataset'] = pd.Categorical(
        read_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    
    read_df = read_df.sort_values('Dataset')
    ax_read = read_df.plot(
        x='Dataset', 
        kind='bar', 
        figsize=(15, 8), 
        width=0.8,
        color=colors[:len(read_df.columns) - 1]  # Ensure that colors match the number of columns
    )
    ax_read.set_title(f'Readability metrics for EU datasets')
    ax_read.set_xlabel('Dataset Name')
    ax_read.set_ylabel('Percentage')
    ax_read.set_xticklabels(read_df['Dataset'], rotation=45, ha="right")
    ax_read.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path_lexical = os.path.join(output_folder, f'read_metrics_eu.png')
    plt.savefig(plot_path_lexical, bbox_inches='tight')
    read_df.to_csv(os.path.join(output_folder, f'read_metrics_eu.csv'))



    # # ACOUSTIC
    combined_acoust_df = pd.concat(all_acoust_dfs, ignore_index=True)
    combined_acoust_df = combined_acoust_df.drop(columns=columns_to_drop, errors='ignore')
    combined_acoust_df = combined_acoust_df.drop(columns='Min duration', errors='ignore')
    combined_acoust_df = combined_acoust_df.rename(columns=columns_to_change)
    
    combined_acoust_df['Dataset'] = pd.Categorical(
        combined_acoust_df['Dataset'], 
        categories=desire_dataset_name, 
        ordered=True
    )
    combined_acoust_df = combined_acoust_df.sort_values('Dataset')

    print(combined_acoust_df)
        
    df_time = combined_acoust_df[['Avg duration', 'Duration std', 'Dataset']]
    df_pauses = combined_acoust_df[['Avg pauses per audio file', 'Avg pause duration', 'Pause duration std', 'Dataset']]
    df_pitch_sample = combined_acoust_df[['Avg pitch', 'Pitch std', 'Sample rate', 'Sample rate std', 'Dataset']]
    df_volume = combined_acoust_df[['Avg volume', 'Volume std', 'Dataset']]
    print(df_volume)
    # exit()

    plot_time_and_pauses(df_time, df_pauses, output_folder)
    plot_pitch_and_sample_rate(df_pitch_sample, output_folder)
    plot_volume(df_volume, output_folder)



def plot_volume(df_volume, output_folder):
    colors = ['purple']
    fig, ax_volume = plt.subplots(figsize=(10, 6))

    # Convert volume values to negative numbers
    avg_volume = -df_volume['Avg volume']
    volume_std = df_volume['Volume std']
    
    lower_error = avg_volume - volume_std
    upper_error = avg_volume + volume_std
    
    # Plot a line with markers instead of bars
    ax_volume.plot(df_volume['Dataset'], avg_volume, color=colors[0], marker='o', linestyle='-', label='Avg Volume')
    ax_volume.fill_between(df_volume['Dataset'], lower_error, upper_error, color=colors[0], alpha=0.3, label='Volume std')

    ax_volume.set_xlabel('Dataset Name')
    ax_volume.set_ylabel('Avg Volume (dB)', color=colors[0])
    ax_volume.set_title('Volume Metrics for EU Datasets')
    ax_volume.tick_params(axis='y', labelcolor=colors[0])
    ax_volume.set_xticklabels(df_volume['Dataset'], rotation=45, ha="right")

    ax_volume.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot
    plot_path_volume = os.path.join(output_folder, 'volume_metrics.png')
    plt.savefig(plot_path_volume, bbox_inches='tight')
    plt.close()

    # Save the dataframe as CSV
    df_volume.to_csv(os.path.join(output_folder, 'volume_metrics.csv'), index=False)




def plot_time_and_pauses(df_time, df_pauses, output_folder):
    colors = ['blue', 'orange', 'green', 'red']
    fig, ax_time = plt.subplots(figsize=(15, 8))

    avg_duration = df_time['Avg duration']
    duration_std = df_time['Duration std']
    
    lower_error = avg_duration - duration_std
    upper_error = avg_duration + duration_std
    
    ax_time.bar(df_time['Dataset'], avg_duration, color=colors[0], width=0.4, label='Avg duration')
    ax_time.errorbar(df_time['Dataset'], avg_duration, yerr=[avg_duration - lower_error, upper_error - avg_duration], fmt='o', color='black', capsize=5, label='Duration std')

    ax_time.set_xlabel('Dataset Name')
    ax_time.set_ylabel('Avg Duration (seconds)', color=colors[0])
    ax_time.set_title('Time and Pause Metrics over the EU Datasets')
    ax_time.tick_params(axis='y', labelcolor=colors[0])
    ax_time.set_xticklabels(df_time['Dataset'], rotation=45, ha="right")
    
    ax_pauses = ax_time.twinx()

    # plot the pause metrics
    ax_pauses.plot(df_pauses['Dataset'], df_pauses['Avg pauses per audio file'], color=colors[1], marker='o', label='Avg pauses per audio file')
    ax_pauses.plot(df_pauses['Dataset'], df_pauses['Avg pause duration'], color=colors[2], marker='s', label='Avg pause duration')
    ax_pauses.fill_between(df_pauses['Dataset'], df_pauses['Avg pause duration'] - df_pauses['Pause duration std'], df_pauses['Avg pause duration'] + df_pauses['Pause duration std'], color=colors[2], alpha=0.3, label='Pause duration std')

    ax_pauses.set_ylabel('Pause Metrics (seconds)', color=colors[1])
    ax_pauses.tick_params(axis='y', labelcolor=colors[1])
    ax_time.legend(loc='upper left', bbox_to_anchor=(0.25, 1))
    ax_pauses.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_acoust = os.path.join(output_folder, 'time_pauses_metrics_eu.png')
    plt.savefig(plot_path_acoust, bbox_inches='tight')
    plt.close()

    # save
    df_time.to_csv(os.path.join(output_folder, 'duration_metrics_eu.csv'), index=False)
    df_pauses.to_csv(os.path.join(output_folder, 'pauses_metrics_eu.csv'), index=False)


def plot_pitch_and_sample_rate(df_pitch_sample, output_folder):
    colors = ['green', 'orange']
    fig, ax_pitch = plt.subplots(figsize=(15, 8))

    avg_pitch = df_pitch_sample['Avg pitch']
    pitch_std = df_pitch_sample['Pitch std']

    ax_pitch.bar(df_pitch_sample['Dataset'], avg_pitch, color=colors[0], width=0.4, label='Avg pitch')
    ax_pitch.errorbar(df_pitch_sample['Dataset'], avg_pitch, yerr=pitch_std, fmt='o', color='black', capsize=5, label='Pitch std')
    ax_pitch.set_xlabel('Dataset Name')
    ax_pitch.set_ylabel('Pitch (Hz)', color=colors[0])
    ax_pitch.set_title('Pitch and Sample Rate Metrics over the EU Datasets')
    ax_pitch.tick_params(axis='y', labelcolor=colors[0])
    ax_pitch.set_xticklabels(df_pitch_sample['Dataset'], rotation=45, ha="right")

    ax_sample_rate = ax_pitch.twinx()

    avg_sample_rate = df_pitch_sample['Sample rate']
    sample_rate_std = df_pitch_sample['Sample rate std']

    ax_sample_rate.plot(df_pitch_sample['Dataset'], avg_sample_rate, color=colors[1], marker='s', label='Avg Sample Rate')
    ax_sample_rate.fill_between(df_pitch_sample['Dataset'], np.maximum(0, avg_sample_rate - sample_rate_std), avg_sample_rate + sample_rate_std, color=colors[1], alpha=0.3, label='Sample Rate std')
    ax_sample_rate.set_ylabel('Sample Rate (Hz)', color=colors[1])
    ax_sample_rate.tick_params(axis='y', labelcolor=colors[1])
    ax_pitch.legend(loc='upper left', bbox_to_anchor=(0.25, 1))
    ax_sample_rate.legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path_acoust = os.path.join(output_folder, 'pitch_sample_rate_metrics_eu.png')
    plt.savefig(plot_path_acoust, bbox_inches='tight')
    plt.close()

    df_pitch_sample.to_csv(os.path.join(output_folder, 'pitch_sample_rate_metrics_eu.csv'), index=False)


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