import pandas as pd
import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from lexical_diversity import lex_div as ld
from tqdm import tqdm
import math
import numpy as np

nltk.download('punkt')

# Load the Spanish model in spacy
nlp = spacy.load('es_core_news_sm')
nlp.max_length = 2000000



#### FUNCTIONS TO PROCESS LINGUISTICS METRICS ####
def extract_references_csv(log_file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(log_file_path)
        df['transcript'] = df['transcript'].astype(str)
        
        # remove empty and NaN values from 'transcript' column
        references = df['transcript'].dropna().tolist()
        references = [ref for ref in references if ref.strip()]  # remove empty strings

    except:
        df = pd.read_csv(log_file_path, on_bad_lines='skip')
        df['transcript'] = df['transcript'].astype(str)
        references = df['transcript'].dropna().tolist()
        references = [ref for ref in references if ref.strip()] # remove empty strings
    return references



def analysis_per_sentence(text,entries, ttr_threshold=0.72):
    doc = nlp(text)
    words = word_tokenize(text, language='spanish')
    sentences = sent_tokenize(text, language='spanish')

    
    # Counting lexical 
    words_count = len(words)

    return {
        "entries": entries,
        "reference": text,
        "words_count": words_count,
    }



def text_analysis(all_statistics):
    total_words = 0
    total_sentences = 0
    
    # List to store words per sentence for calculating the standard deviation
    sentence_lengths = []

    # Sum up all metrics from all statistics
    for stats in all_statistics:
        sentences = sent_tokenize(stats['reference'])
        sentence_count = len(sentences)
        total_sentences += sentence_count

        for sentence in sentences:
            word_count = len(word_tokenize(sentence, language='spanish'))
            total_words += word_count
            sentence_lengths.append(word_count)


    std_dev_words_per_sentence = np.std(sentence_lengths)
    print(f"std_dev_words_per_sentence: {std_dev_words_per_sentence}\n")
    # exit()
    


def process_csv_files(directory):
    files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('_transcriptions.csv')]
    all_statistics = []
    print()

    for file in files:
        output_path = os.path.dirname(file)
        # print(f"Output path -> {output_path}")

        references_list = extract_references_csv(file)
        entries = len(references_list)

        for reference in references_list:
            stats = analysis_per_sentence(reference, entries)
            all_statistics.append(stats)
    return all_statistics, output_path


def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]

    for subdirectory in tqdm(subdirectories, desc=f"Processing directories"):
        if "comparative" in subdirectory or "test" in subdirectory:
            continue

        print(f"Processing directory: {subdirectory}")
        all_stats, output_path = process_csv_files(subdirectory)
        # print(f"Final output path for {subdirectory} --> {output_path}")
        analisys_data = text_analysis(all_stats)


if __name__ == "__main__":
    main()