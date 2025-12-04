import os
import re
import pandas as pd
from tqdm import tqdm
from num2words import num2words


def convert_numbers_to_words(text, lang='es'):
    # Regex to find numbers, \b ensures we select whole numbers
    return re.sub(r'\b\d+\b', lambda x: num2words(x.group(), lang=lang), text)

def clean_text(text):
    text = re.sub(r'ã', 'á', text)
    text = re.sub(r'ï', '', text)
    text = re.sub(r'ã©', 'é', text)
    text = re.sub(r'ã³', 'ó', text)
    text = re.sub(r'á³', 'ó', text)
    text = re.sub(r'á±', 'ñ', text)
    text = re.sub(r'ã', 'í', text)
    text = re.sub(r'á­', 'í', text)
    text = re.sub(r'á©', 'e', text)
    return text

def clean_transcriptions(transcription):
    transcription = transcription.replace("¤", "").replace("=", "").replace("xxx", "").replace("hhh", "").replace("&", "").replace("¿ ?", "").replace("-", "").replace("[///]", "").replace("<", "").replace("[<]", "").replace("+", "").replace("//", "").replace("¿?", "").replace("/", "").replace("¬", "").replace("(%com: everybody laughs)", "").replace("%", "").replace("{act: everybody laughs)", "").replace("[]", "")
    alt_pattern = re.compile(r'{%.*?%}')
    transcription = re.sub(r'\|.*?\|', '', transcription)
    transcription = re.sub(alt_pattern, '', transcription)
    comment_pattern = re.compile(r'\{.*?\}')
    transcription = re.sub(comment_pattern, '', transcription)
    transcription = transcription.replace(" / ", " ").replace(" // ", " ").replace("[/]", "")
    transcription = transcription.replace(">", "").replace("eh", "").replace("mm", "").replace("&eh", "").replace("ah e a", "")
    transcription = " ".join(transcription.split()).strip()
    transcription = transcription.lstrip()
    return transcription


def save_to_csv(file_path, references, hypotheses, audio_files):
    df = pd.DataFrame({
        'path': audio_files,
        'transcript': references,
        'hypothesis': hypotheses
    })
    df.to_csv(file_path, index=False)

# Extract references, hypotheses, and corresponding audio files from log files
def extract_references_and_hypotheses(log_file_path):
    references = []
    hypotheses = []
    audio_files = []
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        current_audio_file = None
        for line in lines:
            # Assuming the audio file path is indicated in the log file
            audio_match = re.search(r'Audio file:\s+(.*)', line)
            if audio_match:
                current_audio_file = audio_match.group(1).strip()

            ref_match = re.search(r'Reference:\s+(.*)', line)
            hyp_match = re.search(r'Hypothesis:\s+(.*)', line)
            
            if ref_match:
                reference = ref_match.group(1).strip()
                references.append(reference)
                audio_files.append(current_audio_file)
            
            if hyp_match:
                hypothesis = hyp_match.group(1).strip()
                hypotheses.append(hypothesis)
    
    return references, hypotheses, audio_files

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_log_files_in_directory(directory):
    ensure_directory(os.path.join(directory, 'linguistics'))  # Ensure the 'linguistics' folder exists
    files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith('.log') and file != 'processing_logs.log']
    
    for file in tqdm(files, desc="Processing log files"):
        linguistics_path = os.path.join(os.path.dirname(file), 'linguistics')
        ensure_directory(linguistics_path)  # Ensure the 'linguistics' folder exists for each subdirectory
        
        output_file_path = os.path.join(linguistics_path, f'{os.path.splitext(os.path.basename(file))[0]}_references_hypotheses.csv')
        
        references, hypotheses, audio_files = extract_references_and_hypotheses(file)
        
        # Save to CSV using the new save_to_csv function
        save_to_csv(output_file_path, references, hypotheses, audio_files)

def main():
    base_directory = os.getcwd()
    process_log_files_in_directory(base_directory)

if __name__ == "__main__":
    main()
