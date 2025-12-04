from num2words import num2words
import glob
import argparse
import pandas as pd
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import codecs
import json
import time



def load_data_common_voice(text_path, audio_path, output_folder, database):
    df = pd.read_csv(text_path, sep='\t')   
    # Save txt file
    sentences = df['sentence'].dropna().tolist()
    
    # Create full paths for the wav files
    df['path'] = df['path'].apply(lambda x: os.path.join(audio_path, f"{x}"))
    df = df[['path', 'sentence']].dropna()
    df.rename(columns={'sentence': 'transcript'}, inplace=True)
    
    # Save the collected data using the utility function
    save_transcription_data(df, sentences, output_folder, database)



def load_data_120h(text_path, audio_path, output_folder, database):
    df = pd.read_csv(text_path)
    
    # Collect sentences
    sentences = df['transcript'].dropna().tolist()
    
    # Create full paths for the wav files
    df['path'] = df['wav_filename'].apply(lambda x: os.path.join(audio_path, f"{x}"))
    df = df[['path', 'transcript']].dropna()

    # Save the collected data using the utility function
    save_transcription_data(df, sentences, output_folder, database)




def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_data = []

    for unit in root.findall(".//UNIT"):
        start_time = float(unit.attrib["startTime"])
        end_time = float(unit.attrib["endTime"])
        transcription = unit.text if unit.text else ""  # none type
        transcription = transcription.strip()
        all_data.append({
            'transcript': transcription,
            'start_time': start_time,
            'end_time': end_time
        })
    return all_data

def clean_transcriptions(transcription):
    transcription = transcription.replace("¤", "").replace("=", "").replace("xxx", "").replace("hhh", "").replace("&", "").replace("¿ ?", "").replace("-", "").replace("[///]", "").replace("<", "").replace("[<]", "").replace("+", "").replace("//", "").replace("¿?", "").replace("/", "").replace("¬", "").replace("(%com: everybody laughs)", "").replace("%", "").replace("{act: everybody laughs)", "").replace("[]", "")
    alt_pattern = re.compile(r'{%.*?%}')
    transcription = re.sub(alt_pattern, '', transcription)
    comment_pattern = re.compile(r'\{.*?\}')
    transcription = re.sub(comment_pattern, '', transcription)
    transcription = transcription.replace(" / ", " ").replace(" // ", " ").replace("[/]", "")
    transcription = transcription.replace(">", "").replace("eh", "").replace("mm", "").replace("&eh", "").replace("ah e a", "")
    transcription = " ".join(transcription.split()).strip()
    transcription = transcription.lstrip()
    return transcription

def load_data_albayzin(prompt_path, wave_path, output_folder, database):
    all_data = []
    
    for xml_file in os.listdir(prompt_path):
        if xml_file.endswith('.xml'):
            parsed_data = parse_xml(os.path.join(prompt_path, xml_file))
            for entry in parsed_data:
                entry['transcript'] = clean_transcriptions(entry['transcript'])
                entry['duration'] = entry['end_time'] - entry['start_time']
                full_audio_path = os.path.join(wave_path, xml_file.replace('.xml', '.wav'))
                if os.path.exists(full_audio_path):
                    entry['wav_filename'] = full_audio_path
                    all_data.append(entry)

    df = pd.DataFrame(all_data)
    sentences = df['transcript'].dropna().tolist()
    
    # Save the collected data using the utility function
    save_transcription_data(df, sentences, output_folder, database)


def load_data_ASR_SpCSC(prompt_path, wave_path, output_folder, database):
    all_data = []
    for txt_file in os.listdir(prompt_path):
        with codecs.open(os.path.join(prompt_path, txt_file), 'r', encoding='utf-8-sig') as file:
            
            raw_lines = file.readlines()
            for line in raw_lines:
                parts = line.strip().split()
                
                if len(parts) >= 4 and parts[1].startswith("G"):
                    start_time = float(parts[0].replace('[', '').split(',')[0])
                    end_time = float(parts[0].replace(']', '').split(',')[1])
                    transcription = " ".join(parts[3:])
                    transcription = transcription.replace("mmm...", "").replace("eh,", "").replace("eh", "").replace("uhm,", "").replace("uhm", "").replace("ja,", "").replace("ja", "").replace("+", "")
                    full_audio_path = os.path.join(wave_path, txt_file.replace('.txt', '.wav'))
                    
                    if 'voice data collection for Beijing Magic Data' in transcription:
                        # remove the entry
                        continue
                    # exit()
                    
                    if os.path.exists(full_audio_path):
                        all_data.append({
                            'wav_filename': full_audio_path,
                            'transcript': transcription,
                            'start_time': start_time,
                            'end_time': end_time
                        })

    df = pd.DataFrame(all_data)
    sentences = df['transcript'].dropna().tolist()

    save_transcription_data(df, sentences, output_folder, database)



def load_data_Europarl_TS(prompt_path, audio_list_path, audio_path, output_folder, database):
    # reading data
    with open(prompt_path, "r") as file:
        transcriptions = file.readlines()
    with open(audio_list_path, "r") as file:
        base_filenames = [filename.strip() for filename in file.readlines()]
    
    # Create full audio filenames with paths
    audio_filenames = []
    for base_filename in base_filenames:
        audio_filenames.append(os.path.join(audio_path, f"{base_filename}.m4a"))
    
    data = {
        'wav_filename': audio_filenames,
        'transcript': [transcription.strip() for transcription in transcriptions]
    }

    df = pd.DataFrame(data)
    sentences = df['transcript'].dropna().tolist()
    
    save_transcription_data(df, sentences, output_folder, database)


  


def load_data_King_ASR(prompt_path, wave_path, output_folder,database):
    all_data = []
    for txt_file in os.listdir(prompt_path):
        
        with codecs.open(os.path.join(prompt_path, txt_file), 'r', encoding='utf-8-sig') as file:
            raw_lines = file.readlines()
            lines = [' '.join(line.strip().split()) for line in raw_lines]
            
            for i in range(1, len(lines)):  
                if "C1" in lines[i] and lines[i-1].split()[0].isdigit():
                    file_name = lines[i-1].split()[0]
                    transcription = lines[i].split("C1")[1].strip().replace('<NON/>', '').replace('<SPK/>', '').replace('<FIL/>', '').replace('**', '')
                    full_audio_path = os.path.join(wave_path, txt_file.replace('.txt', ''), file_name + '.wav')
                    
                    if os.path.exists(full_audio_path) and transcription:  # Checks if transcription is not empty
                        all_data.append({
                            'wav_filename': full_audio_path,
                            'transcript': transcription
                        })
                
                else:
                    continue

    df = pd.DataFrame(all_data)
    sentences = df['transcript'].dropna().tolist()

    save_transcription_data(df, sentences, output_folder, database)




def load_data_MintzAI_ST(prompt_path, wave_path, output_folder, database):
    wav_files = [file for file in os.listdir(wave_path) if file.endswith('.m4a')]
    if not wav_files:
        print("No audio file found in the directory.")
        raise ValueError("No audio file found in the directory.")

    transcripts = []
    for wav_file in wav_files:
        txt_filename = wav_file.replace('.m4a', '.es')
        txt_filepath = os.path.join(prompt_path, txt_filename)
        
        if not os.path.exists(txt_filepath):
            print(f"Transcription file {txt_filepath} does not exist.")
            raise ValueError(f"Transcription file {txt_filepath} does not exist.")

        with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
            transcript = txt_file.readline().strip()
            transcripts.append(transcript)

    if len(wav_files) != len(transcripts):
        print("The number of lines in the txt file doesn't match the number of wav files.")
        raise ValueError("The number of lines in the txt file doesn't match the number of wav files.")

    # Create a DataFrame with full paths for the audio files
    df = pd.DataFrame({
        'wav_filename': [os.path.join(wave_path, wav_file) for wav_file in wav_files],
        'transcript': transcripts
    })

    sentences = df['transcript'].dropna().tolist()
    
    save_transcription_data(df, sentences, output_folder, database)





def load_data_OpenSLR(directory, output_folder, database):
    root_dir = "/mnt/corpus/OpenSLR/SLR108_ES/ES/"
    flac_files = glob.glob(os.path.join(directory, '*.flac'))

    txt_files = [os.path.join(root_dir, os.path.basename(os.path.splitext(audio_file)[0]) + '.txt') for audio_file in flac_files]
    file_pairs = list(zip(flac_files, txt_files))
    
    references = []

    for fun, (audio_file, txt_file) in enumerate(file_pairs):
        if not os.path.exists(txt_file):
            print(f"Text file {txt_file} does not exist. Skipping audio {audio_file}.")
            continue
        
        with open(txt_file, 'r') as f:
            reference = f.read().strip()
            references.append({'wav_filename': audio_file, 'transcript': reference})
    
    df = pd.DataFrame(references)
    sentences = df['transcript'].dropna().tolist()

    save_transcription_data(df, sentences, output_folder, database)
    


def load_data_M_AILABS(text_path, output_folder, database):
    all_data = []
    all_sentences = []

    for root, dirs, files in os.walk(text_path):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                with open(json_path) as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data).T
                df.reset_index(inplace=True)
                df.columns = ['path', 'original', 'clean']
                df = df[['path', 'clean']]
                df.rename(columns={'clean': 'transcript'}, inplace=True)

                df['path'] = df['path'].apply(lambda x: os.path.join(root, 'wavs', x))
                all_data.extend(df.values.tolist())
                all_sentences.extend(df['transcript'].dropna().tolist())

    all_df = pd.DataFrame(all_data, columns=['path', 'transcript'])

    save_transcription_data(all_df, all_sentences, output_folder, database)





def replace_numbers_in_text(text):
    for num in re.findall(r'\b\d+\b', text):
        text = text.replace(num, num2words(num, lang='es'), 1)
    return text

def load_data_parlamento_EJ(text_path, output_folder, database):
    all_data = []
    all_sentences = []

    for root, dirs, files in os.walk(text_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            wav_files = sorted([file for file in os.listdir(dir_path) if file.endswith('.wav') and 'k_' in file])
            txt_file = next((file for file in os.listdir(dir_path) if file.endswith('.txt')), None)
            if txt_file is None:
                continue
            
            with open(os.path.join(dir_path, txt_file), 'r') as file:
                lines = file.readlines()
                transcripts = [re.sub(r' \d+(\.\d+)? \d+(\.\d+)?$', '', line).strip() for line in lines]
                transcripts = [replace_numbers_in_text(transcript) for transcript in transcripts]

            if len(transcripts) != len(wav_files):
                print(f"Warning: The number of lines in the txt file doesn't match the number of wav files in {dir_path}.")
                continue

            wav_files_full_path = [os.path.join(dir_path, file) for file in wav_files]
            for wav_file, transcript in zip(wav_files_full_path, transcripts):
                all_data.append((wav_file, transcript))
                all_sentences.append(transcript)

    df = pd.DataFrame(all_data, columns=['wav_filename', 'transcript'])
    
    save_transcription_data(df, all_sentences, output_folder, database)




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

def load_data_TTS_DB(text_path, output_folder, database):
    all_data = []
    all_sentences = []

    for root, dirs, files in os.walk(text_path):
        for dir_name in dirs:
            if not dir_name.endswith("_es"):
                continue

            dir_path = os.path.join(text_path, dir_name)
            
            if dir_name == 'urkullu_es':
                print("urkullu")
                wav_files = sorted([file for file in os.listdir(dir_path) if file.endswith('.wav')])
                transcripts = []
                for wav_file in wav_files:
                    txt_filename = wav_file.replace('.wav', '.txt')
                    txt_filepath = os.path.join(dir_path, txt_filename)

                    if not os.path.exists(txt_filepath):
                        print(f"Transcription file {txt_filepath} does not exist.")
                        continue

                    with open(txt_filepath, 'r', encoding='ISO-8859-15') as txt_file:
                        transcript = txt_file.readline().strip()
                        transcript = clean_text(transcript)
                        transcript = replace_numbers_in_text(transcript)
                        transcripts.append(transcript)

                if len(transcripts) != len(wav_files):
                    print(f"Warning: The number of txt files doesn't match the number of wav files in {dir_name}.")
                    continue

                wav_files_full_path = [os.path.join(dir_path, file) for file in wav_files]
            else:
                wav_path = os.path.join(dir_path, 'wav')
                txt_path = os.path.join(dir_path, 'txt')
                print(wav_path)
                print(txt_path)

                wav_files = sorted([file for file in os.listdir(wav_path) if file.endswith('.wav')])
                transcripts = []
                unique_files = set()

                for wav_file in wav_files:
                    base_filename = re.sub(r'_\d+', '', wav_file.replace('.wav', ''))
                    if base_filename in unique_files:
                        continue
                    unique_files.add(base_filename)

                    txt_filename = f"{base_filename}.txt"
                    txt_filepath = os.path.join(txt_path, txt_filename)

                    if not os.path.exists(txt_filepath):
                        print(f"Transcription file {txt_filepath} does not exist.")
                        continue

                    with open(txt_filepath, 'r', encoding='ISO-8859-15') as txt_file:
                        transcript = txt_file.readline().strip()
                        transcript = clean_text(transcript)
                        transcript = replace_numbers_in_text(transcript)
                        transcripts.append((os.path.join(wav_path, wav_file), transcript))

                wav_files_full_path, transcripts = zip(*transcripts)

            for wav_file, transcript in zip(wav_files_full_path, transcripts):
                all_data.append((wav_file, transcript))
                all_sentences.append(transcript)

    df = pd.DataFrame(all_data, columns=['wav_filename', 'transcript'])
    
    save_transcription_data(df, all_sentences, output_folder, database)
        


def save_transcription_data(df, all_sentences, output_folder, database):
    # Save all the sentences from the transcript column in a txt file
    sentences_file_path = os.path.join(output_folder, f"{database}_raw_transcription.txt")
    with open(sentences_file_path, 'w', encoding='utf-8') as file:
        for sentence in all_sentences:
            file.write(f"{sentence}\n")
    print(f"Sentences saved to {sentences_file_path}")
    
    # Save the DataFrame with full paths as a CSV file
    csv_file_path = os.path.join(output_folder, f"{database}_transcriptions.csv")
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"Transcriptions and audio file paths saved to {csv_file_path}\n")



def save_output_dir(current_path, database):
    database_name = database.replace(".0", "")
    output_folder = os.path.join(current_path, database_name, 'linguistics', 'raw_data')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created directory {output_folder}")
    else:
        print(f"Directory {output_folder} exists.")

    return output_folder



def process_databases():
    current_path = os.getcwd()
    databases = [
        {"text_path": "/mnt/corpus/Common_Voice_v9.0/es/test.tsv", "audio_path":"/mnt/corpus/Common_Voice_v9.0/es/clips/"},
        {"text_path": "/mnt/corpus/Common_Voice_v12.0/es/test.tsv", "audio_path":"/mnt/corpus/Common_Voice_v12.0/es/clips/"},
        {"text_path": "/mnt/corpus/Common_Voice_v15.0/es/test.tsv", "audio_path":"/mnt/corpus/Common_Voice_v15.0/es/clips/"},
        {"text_path": "/mnt/corpus/120h_Spanish_Speech/asr-spanish-v1-carlfm01/files.csv", "audio_path":"/mnt/corpus/120h_Spanish_Speech/asr-spanish-v1-carlfm01/"},
        {"text_path": "/mnt/corpus/ALBAYZIN2016_ASR/training/transcription", "audio_path": "/mnt/corpus/ALBAYZIN2016_ASR/training/audio"},
        {"text_path": "/mnt/corpus/ASR-SpCSC/TXT/", "audio_path": "/mnt/corpus/ASR-SpCSC/WAV/"},
        {"text_path": "/mnt/corpus/Europarl-ST/v1.1/es/en/test/speeches.es", "audio_path": "/mnt/corpus/Europarl-ST/v1.1/es/audios", "audio_list": "/mnt/corpus/Europarl-ST/v1.1/es/en/test/speeches.lst"},
        {"text_path": "/mnt/corpus/King-ASR-L-202/prompt/", "audio_path":"/mnt/corpus/King-ASR-L-202/wave/"},
        {"text_path": "/mnt/corpus/MintzAI-ST/v1.0/es-eu/test/transcriptions", "audio_path": "/mnt/corpus/MintzAI-ST/v1.0/es-eu/test/audio/"},
        {"text_path": "/mnt/corpus/OpenSLR/SLR108_ES/ES", "audio_path": "/mnt/corpus/OpenSLR/SLR108_ES/ES"},
        {"text_path": "/mnt/corpus/M-AILABS/"},
        {"text_path": "/mnt/corpus/Parlamento_EJ/ES"},
        {"text_path": "/mnt/corpus/TTS_DB/"},
    ]


    for db in databases:
        text_path = Path(db.get("text_path", ""))
        database = text_path.parts[3]


        if "Common_Voice" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_common_voice(db["text_path"], db["audio_path"], output_folder, database)


        elif "120h_Spanish_Speech" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_120h(db["text_path"], db["audio_path"], output_folder, database)


        elif "ALBAYZIN2016_ASR" in database:
            database = "ALBAYZIN2016_ASR"
            output_folder = save_output_dir(current_path, database)
            load_data_albayzin(db["text_path"], db["audio_path"], output_folder, database)


        elif "ASR-SpCSC" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_ASR_SpCSC(db["text_path"], db["audio_path"], output_folder, database)


        elif "Europarl-ST" in database:
            database = "Europarl-TS"
            output_folder = save_output_dir(current_path, database)
            load_data_Europarl_TS(db["text_path"], db["audio_list"], db["audio_path"], output_folder, database)


        elif "King-ASR-L-202" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_King_ASR(db["text_path"], db["audio_path"], output_folder, database)


        elif "MintzAI-ST" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_MintzAI_ST(db["text_path"], db["audio_path"], output_folder, database)


        elif "OpenSLR" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_OpenSLR(db["audio_path"], output_folder, database)


        elif "M-AILABS" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_M_AILABS(db["text_path"], output_folder, database)


        elif "Parlamento_EJ" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_parlamento_EJ(db["text_path"], output_folder, database)


        elif "TTS_DB" in database:
            output_folder = save_output_dir(current_path, database)
            load_data_TTS_DB(db["text_path"], output_folder, database)

        else:
            print(f"Unknown database format")


def main():
    parser = argparse.ArgumentParser(description="Process all specified databases.")
    args = parser.parse_args()
    process_databases()


if __name__ == "__main__":
    main()
