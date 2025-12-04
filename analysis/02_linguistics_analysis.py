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


def syllable_count(word):
    return sum(char in 'aeiouáéíóúü' for char in word.lower())



def calculate_dependency_metrics(doc):
    total_depth = 0
    total_clauses = 0
    metrics_per_sentence = [] 
    
    for sent in doc.sents:
        sentence_depth = 0
        clauses = 0
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            sentence_depth += depth
        
        clauses = sum(1 for token in sent if token.dep_ in {'csubj', 'csubjpass', 'ccomp', 'xcomp'})
        total_clauses += clauses
        total_depth += sentence_depth
        metrics_per_sentence.append({'depth': sentence_depth, 'clauses': clauses})
    return metrics_per_sentence 
    

def calculate_t_units(doc):
    t_units = []
    current_t_unit = []

    for sent in doc.sents:
        current_t_unit = []  # Reset for each new sentence
        for token in sent:
            # Add token to the current T-unit
            current_t_unit.append(token)
            # If token is a ROOT or starts a new clause, conclude current T-unit
            if token.dep_ in {'ccomp', 'xcomp', 'advcl'} and token.head.dep_ == 'ROOT':
                t_units.append(current_t_unit)
                current_t_unit = [token]  # Start new T-unit with the clause initiator
        # Ensure the last collected T-unit in the sentence is added
        if current_t_unit:
            t_units.append(current_t_unit)

    # Calculate the lengths of the T-units
    t_unit_lengths = [len(t_unit) for t_unit in t_units]
    return t_unit_lengths



# Calculate text statistics
def analysis_per_sentence(text, mtld,entries, ttr_threshold=0.72):
    # print(text)
    #Convert to doc spacy
    doc = nlp(text)
    # print(doc)
    # displacy.serve(doc, style="dep")

    # GRammar analysis
    content_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]

    pos_counts = doc.count_by(spacy.attrs.POS)
    noun_count = pos_counts.get(nlp.vocab.strings['NOUN'], 0)
    verb_count = pos_counts.get(nlp.vocab.strings['VERB'], 0)
    adj_count = pos_counts.get(nlp.vocab.strings['ADJ'], 0)
    adv_count = pos_counts.get(nlp.vocab.strings['ADV'], 0)


    # Tokenize text into words and sentences
    words = word_tokenize(text, language='spanish')
    sentences = sent_tokenize(text, language='spanish')

    
    # Counting lexical 
    words_count = len(words)
    total_syllables = sum(syllable_count(word) for word in words)
    complex_words = sum(1 for word in words if syllable_count(word) >= 3)

    char_counts = [len(sentence) for sentence in sentences]
    char_count = sum(char_counts)



    # Calculating syntaxis metrics
    dependency_count = calculate_dependency_metrics(doc)
    t_unit_count = calculate_t_units(doc)

    return {
        "entries": entries,
        "reference": text,
        "words_count": words_count,

        "total_syllables": total_syllables,
        "complex_words": complex_words,
        "char_count": char_count,
        
        "pos_count": pos_counts,
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adj_count": adj_count,
        "adv_count": adv_count,

        "dependency_count": dependency_count,
        "t_unit_count": t_unit_count,

        "mtld": mtld,
    }



def text_analysis(all_statistics):
    total_words = 0
    total_sentences = 0
    total_syllables = 0
    total_complex_words = 0
    total_unique_words = 0
    total_chars = 0

    total_nouns = 0
    total_verbs = 0
    total_adjs = 0
    total_advs = 0

    total_depths = 0
    total_clauses = 0
    total_t_units = 0
    
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
        
        total_syllables += stats['total_syllables']
        total_complex_words += stats['complex_words']
        total_unique_words += len(set(word_tokenize(stats['reference'], language='spanish')))
        total_chars += stats['char_count']

        total_nouns += stats['noun_count']
        total_verbs += stats['verb_count']
        total_adjs += stats['adj_count']
        total_advs += stats['adv_count']

        for dep in stats['dependency_count']:
            total_depths += dep['depth']
            total_clauses += dep['clauses']
        total_t_units += len(stats['t_unit_count'])

    # Calculate averages
    avg_words_per_sentence = total_words / total_sentences
    avg_depths = total_depths / total_sentences
    avg_clauses = total_clauses / total_sentences
    avg_t_units = total_t_units / total_sentences
    total_pos = total_nouns + total_verbs + total_adjs + total_advs
    mtld = stats["mtld"]
    total_entries = stats["entries"]

    # Standard deviation for words per sentence using numpy
    std_dev_words_per_sentence = np.std(sentence_lengths)
    # print(f"Standard deviation: {std_dev_words_per_sentence}")
    # exit()

    # Further calculations
    flesch_ease = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    flesch_kincaid_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    gunning_fog_index = 0.4 * ((total_words / total_sentences) + 100 * (total_complex_words / total_words))
    ari = 4.71 * (total_chars / total_words) + 0.5 * (total_words / total_sentences) - 21.43
    coleman_liau_index = 0.0588 * (total_chars / total_words * 100) - 0.296 * (total_sentences / total_words * 100) - 15.8
    vocabulary_density = (total_nouns + total_verbs + total_adjs + total_advs) / total_words
    rttr = total_unique_words / math.sqrt(total_words)

    all_data = {
        "total_entries": total_entries,

        "total_words": total_words,
        "total_sentences": total_sentences,
        "total_syllables": total_syllables,
        "total_complex_words": total_complex_words,
        "total_unique_words": total_unique_words,
        "total_chars": total_chars,
        "avg_words_per_sentence": avg_words_per_sentence,
        "std_dev_words_per_sentence": std_dev_words_per_sentence,
        
        "vocabulary_density": vocabulary_density,
        "mtld": mtld,
        "rttr": rttr,

        "flesch_ease": flesch_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "gunning_fog_index": gunning_fog_index,
        "ari": ari,
        "coleman_liau_index": coleman_liau_index,

        "total_pos": total_pos,
        "total_nouns": total_nouns,
        "total_verbs": total_verbs,
        "total_adjs": total_adjs,
        "total_advs": total_advs,
        
        "avg_depths": avg_depths,
        "avg_clauses": avg_clauses,
        "avg_t_units": avg_t_units,
    }
    
    return all_data




def calculate_mtld(references_list):
    all_words = []
    for sentence in references_list:
        if isinstance(sentence, str) and sentence.strip():
            words = nltk.word_tokenize(sentence, language='spanish')
            all_words.extend(words)
    
    mtld_score = ld.mtld(all_words)
    return mtld_score


def save_all_data(stats, output_folder):
    folder_name = output_folder.split("/")[-3]

    stats_file_path = os.path.join(output_folder, f'{folder_name}_stats.txt')
    latex_file_path = os.path.join(output_folder, f'{folder_name}_stats.tex')

    # Writing to the text file
    with open(stats_file_path, 'w') as file:
        file.write(f"Processed {folder_name}\n\n")

        file.write(f"Total entries: {stats['total_entries']}\n")
        file.write(f"Total words: {stats['total_words']}\n")
        file.write(f"Unique words: {stats['total_unique_words']}\n")
        file.write(f"Total Sentences: {stats['total_sentences']}\n")
        file.write(f"Mean Words per Sentence: {stats['avg_words_per_sentence']:.2f}\n")
        file.write(f"Standard Deviation of Words per Sentence: {stats['std_dev_words_per_sentence']:.2f}\n\n")  # Added line

        file.write(f"Root Type-Token Ratio (RTTR): {stats['rttr']:.2f}\n")
        file.write(f"Measure of Textual Lexical Diversity (MTLD): {stats['mtld']:.2f}\n")
        file.write(f"Vocabulary Density: {stats['vocabulary_density']:.2f}\n\n")

        file.write(f"Average Parse Tree Depth: {stats['avg_depths']:.2f}\n")
        file.write(f"Average Clauses per Sentence: {stats['avg_clauses']:.2f}\n")
        file.write(f"Average T-Unit Length: {stats['avg_t_units']:.2f}\n\n")

        file.write(f"Flesch Reading Ease: {stats['flesch_ease']:.2f}\n")
        file.write(f"Flesch-Kincaid Grade Level: {stats['flesch_kincaid_grade']:.2f}\n")
        file.write(f"Gunning Fog Index: {stats['gunning_fog_index']:.2f}\n")
        file.write(f"Automated Readability Index (ARI): {stats['ari']:.2f}\n")
        file.write(f"Coleman-Liau Index: {stats['coleman_liau_index']:.2f}\n\n")

        file.write(f"Total PoS: {stats['total_pos']}\n")
        file.write(f"Noun Count: {stats['total_nouns']}\n")
        file.write(f"Verb Count: {stats['total_verbs']}\n")
        file.write(f"Adjective Count: {stats['total_adjs']}\n")
        file.write(f"Adverb Count: {stats['total_advs']}\n")

    # Writing to the LaTeX file
    with open(latex_file_path, 'w') as file:
        file.write(r"\begin{table}[h!]" + "\n")
        file.write(r"    \centering" + "\n")
        file.write(r"    \begin{tabular}{|l|c|}" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \textbf{Metric} & \textbf{Value} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{General Statistics}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Total Entries & {stats['total_entries']} \\\\" + "\n")
        file.write(f"        Total Words & {stats['total_words']} \\\\" + "\n")
        file.write(f"        Total Sentences & {stats['total_sentences']} \\\\" + "\n")
        file.write(f"        Mean Words per Sentence & {stats['avg_words_per_sentence']:.2f} \\\\" + "\n")
        file.write(f"        Standard Deviation of Words per Sentence & {stats['std_dev_words_per_sentence']:.2f} \\\\" + "\n")  # Added line
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Lexical Complexity}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Root Type-Token Ratio (RTTR) & {stats['rttr']:.2f} \\\\" + "\n")
        file.write(f"        Measure of Textual Lexical Diversity (MTLD) & {stats['mtld']:.2f} \\\\" + "\n")
        file.write(f"        Vocabulary Density & {stats['vocabulary_density']:.2f} \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Syntactic Complexity}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Average Parse Tree Depth & {stats['avg_depths']:.2f} \\\\" + "\n")
        file.write(f"        Average Clauses per Sentence & {stats['avg_clauses']:.2f} \\\\" + "\n")
        file.write(f"        Average T-Unit Length & {stats['avg_t_units']:.2f} \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Readability}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Flesch Reading Ease & {stats['flesch_ease']:.2f} \\\\" + "\n")
        file.write(f"        Flesch-Kincaid Grade Level & {stats['flesch_kincaid_grade']:.2f} \\\\" + "\n")
        file.write(f"        Gunning Fog Index & {stats['gunning_fog_index']:.2f} \\\\" + "\n")
        file.write(f"        Automated Readability Index (ARI) & {stats['ari']:.2f} \\\\" + "\n")
        file.write(f"        Coleman-Liau Index & {stats['coleman_liau_index']:.2f} \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{POS Counts}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Total PoS & {stats['total_pos']} \\\\" + "\n")
        file.write(f"        Noun Count & {stats['total_nouns']} \\\\" + "\n")
        file.write(f"        Verb Count & {stats['total_verbs']} \\\\" + "\n")
        file.write(f"        Adjective Count & {stats['total_adjs']} \\\\" + "\n")
        file.write(f"        Adverb Count & {stats['total_advs']} \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"    \end{tabular}" + "\n")
        file.write(r"    \caption{Linguistic Metrics for the Dataset}" + "\n")
        file.write(r"    \label{tab:linguistic_metrics}" + "\n")
        file.write(r"\end{table}" + "\n")

    print(f"Statistics saved to {stats_file_path}")
    print(f"LaTeX table saved to {latex_file_path}\n\n")


def process_csv_files(directory):
    files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('_transcriptions.csv')]
    all_statistics = []
    print()

    for file in files:
        output_path = os.path.dirname(file)
        # print(f"Output path -> {output_path}")

        references_list = extract_references_csv(file)
        mtld = calculate_mtld(references_list)
        entries = len(references_list)

        for reference in references_list:
            stats = analysis_per_sentence(reference, mtld, entries)
            all_statistics.append(stats)
    return all_statistics, output_path


def main():
    base_directory = os.getcwd()
    subdirectories = [os.path.join(base_directory, sub_dir) for sub_dir in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, sub_dir))]

    for subdirectory in tqdm(subdirectories, desc=f"Processing directories"):
        if "comparative" in subdirectory or "test" in subdirectory:
            continue
        # if "Europarl-TS" in subdirectory:
        #     continue
        # if "M-AILABS" in subdirectory:
        #     continue
        # if "MintzAI-ST" in subdirectory:
        #     continue
        # if "Parlamento_EJ" in subdirectory:
        #     continue
        # if "TTS_DB" in subdirectory:
        #     continue
        # if "ALBAYZIN2016_ASR" in subdirectory:
        #     continue
        # if "OpenSLR" in subdirectory:
        #     continue
        # if "King-ASR-L-202" in subdirectory:
        #     continue
        # if "Common_Voice_v9" in subdirectory:
        #     continue
        # if "ASR-SpCSC" in subdirectory:
        #     continue
        # if "Common_Voice_v12" in subdirectory:
        #     continue
        # if "Common_Voice_v15" in subdirectory:
        #     continue
        # if "120h_Spanish_Speech" in subdirectory:
        #     continue

        print(f"Processing directory: {subdirectory}")
        all_stats, output_path = process_csv_files(subdirectory)
        # print(f"Final output path for {subdirectory} --> {output_path}")
        analisys_data = text_analysis(all_stats)

        # save
        # print("Saving data")
        save_all_data(analisys_data, output_path)


if __name__ == "__main__":
    main()