import os
import jiwer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

MODEL_STRING = "DeepSpeech"


def compute_wer_details(reference, hypothesis):
    return jiwer.compute_measures(reference, hypothesis)

def process_files(directory):
    data_for_analysis = []
    high_wer_sentences = []  # To track sentences with WER > 1

    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'references_hypotheses.txt' in file:
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        lines = f.read().strip().split("\n")
                        
                        if len(lines) % 2 != 0:
                            print(f"Skipping file with odd number of lines: {file_path}")
                            continue

                        # Separate references and hypotheses
                        references = lines[::2]  # Even lines (starting from 0) are references
                        hypotheses = lines[1::2]  # Odd lines are hypotheses

                        # Calculate WER for each pair
                        wers = []
                        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                            if not ref.strip():  # Skip if the reference is empty
                                print(f"Skipping empty reference at sentence pair {len(wers) + 1}")
                                continue
                            wer_details = compute_wer_details(ref, hyp)
                            wer = wer_details['wer']
                            if wer > 1:
                                high_wer_sentences.append({
                                    'Reference': ref,
                                    'Hypothesis': hyp,
                                    'Original WER': wer
                                })
                                wer = 1
                            incorrect_words = wer_details['substitutions'] + wer_details['insertions'] + wer_details['deletions']
                            wers.append(wer)
                            
                            data_for_analysis.append({
                                'Reference': ref,
                                'Hypothesis': hyp,
                                
                                'WER': wer,
                                'Words in Reference': len(ref.split()),
                                'Words in Hypothesis': len(hyp.split()),
                                'Incorrect Words': incorrect_words,

                                'Substitutions': wer_details['substitutions'],
                                'Insertions': wer_details['insertions'],
                                'Deletions': wer_details['deletions']
                            })

                        if wers:  # Ensure there are WERs to average
                            average_wer = sum(wers) / len(wers)
                            print(f"\nAverage WER: {average_wer:.4f}")
                        else:
                            print("No valid WERs to calculate an average.")
                
                except Exception as e:
                    print(f"Failed to process file {file_path}: {str(e)}")
    return data_for_analysis, high_wer_sentences



def save_data_and_summarize(data, high_wer_sentences, directory):
    # Create directories
    plot_name = directory.split('/')[-1]
    
    plots_directory = os.path.join(directory, 'WER', 'plots')
    text_directory = os.path.join(directory, 'WER')

    os.makedirs(plots_directory, exist_ok=True)
    os.makedirs(text_directory, exist_ok=True)
    
    # Save the data to CSV in the 'linguistics' directory
    df = pd.DataFrame(data)
    summary_csv_path = os.path.join(text_directory, f'wer_{plot_name}_{MODEL_STRING}.csv')
    summary_data = {
        'Average WER': [df['WER'].mean()],
        'Total Insertions': [df['Insertions'].sum()],
        'Total Deletions': [df['Deletions'].sum()],
        'Total Substitutions': [df['Substitutions'].sum()],
        'Total Words in Reference': [df['Words in Reference'].sum()],
        'Total Words in Hypothesis': [df['Words in Hypothesis'].sum()],
        'Total Incorrect Words': [df['Incorrect Words'].sum()]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary data saved to '{summary_csv_path}'.")

    # Save sentences with original WER > 1 to CSV
    if high_wer_sentences:
        high_wer_csv_path = os.path.join(plots_directory, f'{plot_name}_high_wer_sentences_eu_{MODEL_STRING}.csv')
        pd.DataFrame(high_wer_sentences).to_csv(high_wer_csv_path, index=False)
        print(f"High WER sentences saved to '{high_wer_csv_path}'.")

    # Output average and median WER
    print()
    print(f"Average WER: {df['WER'].mean():.2f}")
    print()

    # PLOTTING RESULTS
    # [1] WER Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['WER'], bins=55, alpha=0.75)
    plt.title(f'Distribution of WER in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('WER')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_histogram_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [2.1] WER vs. Words in Reference
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Words in Reference', y='WER', data=df, color='red')
    plt.title(f'WER vs. Words in Reference in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_reference_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [2.2] bins
    bin_edges = pd.cut(df['Words in Reference'], bins=20)  # Adjust 'bins' to control the number of bins
    df_binned = df.groupby(bin_edges)['WER'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Words in Reference', y='WER', data=df_binned, color='red')
    plt.title(f'WER vs. Words in Reference in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference (Binned)')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(-1, len(df_binned))  # Adjust x-axis limits based on bins
    plt.ylim(0, None)  # Start y-axis at 0
    plt.xticks(rotation=90, ha='right')

    # Format x-axis labels to display integers
    new_labels = [f'{int(bin.left)}-{int(bin.right)}' for bin in df_binned['Words in Reference']]
    plt.gca().set_xticklabels(new_labels)
    plt.tight_layout()

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_reference_binned_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to '{plot_path}'.")


    # [2.3] line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Words in Reference', y='WER', data=df, color='red', marker='o')

    # Set the title and labels
    plt.title(f'WER vs. Words in Reference in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_reference_line_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to '{plot_path}'.")



    # [3.1] WER vs. Incorrect Words
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Incorrect Words', y='WER', data=df, color='green')
    plt.title(f'WER vs. Incorrect Words in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Incorrect Words')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_inc_words_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [3.2] bins
    bin_edges = pd.cut(df['Incorrect Words'], bins=20)  # Adjust 'bins' to control the number of bins
    df_binned = df.groupby(bin_edges)['WER'].mean().reset_index()

    # Plot the binned data
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Incorrect Words', y='WER', data=df_binned, color='green')
    plt.title(f'WER vs. Incorrect Words in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Incorrect Words (Binned)')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(-1, len(df_binned))  # Adjust x-axis limits based on bins
    plt.ylim(0, None)  # Start y-axis at 0
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()

    # Format x-axis labels to display integers
    new_labels = [f'{int(bin.left)}-{int(bin.right)}' for bin in df_binned['Incorrect Words']]
    plt.gca().set_xticklabels(new_labels)
    plt.tight_layout()

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_inc_words_binned_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [3.3] line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Incorrect Words', y='WER', data=df, color='green', marker='o')

    # Set the title and labels
    plt.title(f'WER vs. Incorrect Words in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Incorrect Words')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_inc_words_line_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")



    # [4.1] Words in Reference vs. Incorrect Words
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Words in Reference', y='Incorrect Words', data=df, color='blue')
    plt.title(f'Words in Reference vs. Incorrect Words in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference')
    plt.ylabel('Incorrect Words')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_words_vs_inc_words_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [4.2] kde
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Words in Reference'], label='Words in Reference', color='blue', fill=True)
    sns.kdeplot(df['Incorrect Words'], label='Incorrect Words', color='red', fill=True)
    plt.title(f'Distribution of Words in Reference and Incorrect Words in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference')
    plt.ylabel('Incorrect Words')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_words_vs_inc_words_kde_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [6.1] Words in Hypothesis vs. WER
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Words in Hypothesis', y='WER', data=df, color='purple')
    plt.title(f'WER vs. Words in Hypothesis in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Hypothesis')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_hypothesis_scatter_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [6.2] bins
    bin_edges = pd.cut(df['Words in Hypothesis'], bins=20)  # Adjust 'bins' to control the number of bins
    df_binned = df.groupby(bin_edges)['WER'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Words in Hypothesis', y='WER', data=df_binned, color='red')
    plt.title(f'WER vs. Words in Hypothesis in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Hypothesis (Binned)')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(-1, len(df_binned))  # Adjust x-axis limits based on bins
    plt.ylim(0, None)  # Start y-axis at 0
    plt.xticks(rotation=90, ha='right')

    # Format x-axis labels to display integers
    new_labels = [f'{int(bin.left)}-{int(bin.right)}' for bin in df_binned['Words in Hypothesis']]
    plt.gca().set_xticklabels(new_labels)
    plt.tight_layout()

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_hypothesis_binned_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [6.3] line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Words in Hypothesis', y='WER', data=df, color='red', marker='o')

    # Set the title and labels
    plt.title(f'WER vs. Words in Hypothesis in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Hypothesis')
    plt.ylabel('WER')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis limits
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0

    plot_path = os.path.join(plots_directory, f'{plot_name}_wer_vs_w_hypothesis_line_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [7.1] Words in Reference vs. Words in Hypothesis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Words in Reference', y='Words in Hypothesis', data=df, color='orange')
    plt.title(f'Words in Reference vs. Words in Hypothesis in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Words in Reference')
    plt.ylabel('Words in Hypothesis')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plt.ylim(0, None)  # Start y-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_words_distribution_scatter_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


    # [7.2] Words in Reference vs. Words in Hypothesis
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['Words in Reference'], label='Words in Reference', color='blue', fill=True)
    sns.kdeplot(df['Words in Hypothesis'], label='Words in Hypothesis', color='red', fill=True)
    plt.title(f'Distribution of Words in Reference and Hypothesis in {plot_name} for {MODEL_STRING} model')
    plt.xlabel('Number of Words')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, None)  # Start x-axis at 0
    plot_path = os.path.join(plots_directory, f'{plot_name}_words_distribution_kde_eu_{MODEL_STRING}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to '{plot_path}'.")    


def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]
    
    for directory in tqdm(subdirectories, desc="Processing Directories", unit="dir"):
        if "comparative_analysis" in directory:
            continue
        # if "TTS_DB" in directory:
        print(f"Processing directory: {directory}")
        data, high_wer_sentences = process_files(directory)
        if data:
            save_data_and_summarize(data, high_wer_sentences, directory)
        else:
            print(f"No data collected in {directory}. Check your directory path or file formats.")

if __name__ == "__main__":
    main()
