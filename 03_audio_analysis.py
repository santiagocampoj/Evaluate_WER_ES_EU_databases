import soundfile as sf
import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import parselmouth
from pydub import AudioSegment

MODEL_STRING = "DeepSpeech"

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")


def get_audio_segment(file_path, start_time=None, end_time=None):
    try:
        with sf.SoundFile(file_path) as sound_file:
            start_frame = int(start_time * sound_file.samplerate) if start_time else 0
            end_frame = int(end_time * sound_file.samplerate) if end_time else sound_file.frames
            sound_file.seek(start_frame)
            segment = sound_file.read(end_frame - start_frame)
            rate = sound_file.samplerate
    
    except (RuntimeError, sf.LibsndfileError):
        audio = AudioSegment.from_file(file_path)
    
        if start_time and end_time:
            segment = audio[start_time*1000:end_time*1000]  # milliseconds
        elif start_time:
            segment = audio[start_time*1000:]
        else:
            segment = audio
    
        segment = np.array(segment.get_array_of_samples()).astype(np.float32) / (2**15)
        rate = audio.frame_rate
    return segment, rate


def get_audio_files(directory):
    audio_files = []
    for file in os.listdir(directory):
        if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
            audio_files.append(os.path.join(directory, file))
    return audio_files


def get_audio_duration(segment, rate):
    return len(segment) / float(rate)


def get_audio_sr_volume(segment, rate):
    rms = np.mean(librosa.feature.rms(y=segment))
    volume_db = 20 * np.log10(rms)
    return rate, volume_db


def calculate_snr(segment, rate):
    try:
        signal_power = np.mean(segment**2)
        noise_power = np.var(segment - librosa.effects.hpss(segment)[1])
        snr = 10 * np.log10(signal_power / noise_power)
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        snr = None
    return snr


def get_pause_metrics(segment, rate, top_db, min_pause_duration):
    non_silent_intervals = librosa.effects.split(segment, top_db=top_db)
    pauses = []
    for i in range(1, len(non_silent_intervals)):
        pause = (non_silent_intervals[i][0] - non_silent_intervals[i-1][1]) / rate
        if pause >= min_pause_duration:
            pauses.append(pause)
    total_pauses = len(pauses)
    avg_pause_duration = np.mean(pauses) if pauses else 0
    return total_pauses, avg_pause_duration


def get_pitch_metrics(segment, rate):
    try:
        sound = parselmouth.Sound(segment.astype(np.float64), sampling_frequency=rate)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        # Filtering greater than 50 and less than 350
        pitch_values = pitch_values[(pitch_values > 50) & (pitch_values < 350)]

        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        max_pitch = np.max(pitch_values)
        min_pitch = np.min(pitch_values)
    except Exception as e:
        print(f"Error processing pitch: {e}")
        mean_pitch = std_pitch = max_pitch = min_pitch = 0
    return mean_pitch, std_pitch, max_pitch, min_pitch




def plot_metrics(df, output_folder, database_name):
    plots = [
        ("duration", "Duration (seconds)", "Duration Distribution"),
        ("volume", "Volume (dB)", "Volume Distribution"),
        # ("snr", "SNR (dB)", "SNR Distribution"),
        ("total_pauses", "Total Pauses", "Total Pauses Distribution"),
        ("avg_pause_duration", "Mean Pause Duration (seconds)", "Mean Pause Duration Distribution"),
        ("mean_pitch", "Mean Pitch (Hz)", "Mean Pitch Distribution"),
        ("std_pitch", "Pitch Standard Deviation (Hz)", "Pitch Standard Deviation Distribution"),
    ]

    for column, xlabel, title in plots:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column].dropna())
        plt.title(f'{title} in {database_name} for the {MODEL_STRING} model')
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.xlim(left=0)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(os.path.join(output_folder, f'{database_name}_{column}_distribution_{MODEL_STRING}.png'))
        plt.close()


def analyze_audio_files(df, output_folder, database_name, top_db, min_pause_duration):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if df.empty:
        print("No audio files found in the directory.")
        return
    
    durations = []
    sample_rates = []
    volumes = []
    # snrs = []
    total_pauses_list = []
    avg_pause_durations = []
    mean_pitches = []
    std_pitches = []
    max_pitches = []
    min_pitches = []
    file_paths = []
    start_index = 11777
    # start_index = 11600
    end_index = 11778
    max_df = 5

    # for _, row in tqdm(df.head(max_df).iterrows(), total=min(max_df, df.shape[0]), desc="Processing audio files"):
    # for _, row in tqdm(df.iloc[start_index:end_index].iterrows(), total=(df.shape[0] - start_index), desc="Processing audio files"):
    for _, row in tqdm(df.iterrows(), total=(df.shape[0]), desc="Processing audio files"):
        file_path = row['wav_filename'] if 'wav_filename' in row else row['path']
        start_time = row.get('start_time', None)
        end_time = row.get('end_time', None)
        file_name = file_path.split("/")[-1]
        file_paths.append(file_name)

        try:
            segment, rate = get_audio_segment(file_path, start_time, end_time)
            duration = get_audio_duration(segment, rate)
            durations.append(duration)
            
            rate, volume = get_audio_sr_volume(segment, rate)
            sample_rates.append(rate)
            volumes.append(volume)

            # snr = calculate_snr(segment, rate)
            # snrs.append(snr)

            total_pauses, avg_pause_duration = get_pause_metrics(segment, rate, top_db, min_pause_duration)
            total_pauses_list.append(total_pauses)
            avg_pause_durations.append(avg_pause_duration)

            mean_pitch, std_pitch, max_pitch, min_pitch = get_pitch_metrics(segment, rate)
            mean_pitches.append(mean_pitch)
            std_pitches.append(std_pitch)
            max_pitches.append(max_pitch)
            min_pitches.append(min_pitch)

    
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping...")
            durations.append(np.nan)
            sample_rates.append(np.nan)
            volumes.append(np.nan)
            # snrs.append(np.nan)
            total_pauses_list.append(np.nan)
            avg_pause_durations.append(np.nan)
            mean_pitches.append(np.nan)
            std_pitches.append(np.nan)
            max_pitches.append(np.nan)
            min_pitches.append(np.nan)
            continue

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            durations.append(np.nan)
            sample_rates.append(np.nan)
            volumes.append(np.nan)
            # snrs.append(np.nan)
            total_pauses_list.append(np.nan)
            avg_pause_durations.append(np.nan)
            mean_pitches.append(np.nan)
            std_pitches.append(np.nan)
            max_pitches.append(np.nan)
            min_pitches.append(np.nan)
            continue


    total_files = df.shape[0]
    
    valid_durations = [d for d in durations]
    mean_duration = np.mean(valid_durations)
    max_duration = np.max(valid_durations)
    min_duration = np.min(valid_durations)
    
    valid_sample_rates = [sr for sr in sample_rates]
    mean_sample_rate = np.mean(valid_sample_rates)
    std_sample_rate = np.std(valid_sample_rates)
    
    valid_volumes = [v for v in volumes]
    mean_volume = np.mean(valid_volumes)
    std_volume = np.std(valid_volumes)
    
    valid_avg_pause_durations = [pd for pd in avg_pause_durations]
    mean_pause_duration = np.mean(valid_avg_pause_durations)
    std_pause_duration = np.std(valid_avg_pause_durations)
    
    valid_mean_pitches = [mp for mp in mean_pitches]
    mean_pitch = np.mean(valid_mean_pitches)
    
    valid_std_pitches = [sp for sp in std_pitches]
    std_pitch = np.mean(valid_std_pitches)
    
    valid_max_pitches = [mx for mx in max_pitches]
    max_pitch_value = np.max(valid_max_pitches)
    
    valid_min_pitches = [mn for mn in min_pitches]
    min_pitch_value = np.min(valid_min_pitches)

    avg_pauses_per_audio = (sum(total_pauses_list) / total_files)

    # valid_snrs = [s for s in snrs if s is not None]
    # mean_snr = np.mean(valid_snrs) if valid_snrs else 0
    # std_snr = np.std(valid_snrs) if valid_snrs else 0


    output_text_file = os.path.join(output_folder, f'{database_name}_audio_analysis_{MODEL_STRING}.txt')
    output_csv_file = os.path.join(output_folder, f'{database_name}_audio_analysis_{MODEL_STRING}.csv')
    output_latex_file = os.path.join(output_folder, f'{database_name}_audio_analysis_{MODEL_STRING}.tex')
    
    with open(output_text_file, 'w') as f:
        f.write(f"Total audio files: {total_files}\n")

        f.write(f"Mean duration: {mean_duration:.2f} seconds\n")
        f.write(f"Max duration: {max_duration:.2f} seconds\n")
        f.write(f"Min duration: {min_duration:.2f} seconds\n")

        f.write(f"Mean sample rate: {mean_sample_rate:.2f} Hz\n")
        f.write(f"Sample rate standard deviation: {std_sample_rate:.2f} Hz\n")

        f.write(f"Mean volume: {mean_volume:.2f} dB\n")
        f.write(f"Volume standard deviation: {std_volume:.2f} dB\n")
        # f.write(f"Mean SNR: {mean_snr:.2f} dB\n")
        # f.write(f"SNR standard deviation: {std_snr:.2f} dB\n")

        f.write(f"Total pauses: {sum(total_pauses_list)}\n")
        f.write(f"Mean pauses per audio file: {avg_pauses_per_audio:.2f}\n")
        f.write(f"Mean pause duration: {mean_pause_duration:.2f} seconds\n")
        f.write(f"Pause duration standard deviation: {std_pause_duration:.2f} seconds\n")
        
        f.write(f"Mean pitch: {mean_pitch:.2f} Hz\n")
        f.write(f"Pitch standard deviation: {std_pitch:.2f} Hz\n")
        f.write(f"Max pitch: {max_pitch_value:.2f} Hz\n")
        f.write(f"Min pitch: {min_pitch_value:.2f} Hz\n")
    

    with open(output_latex_file, 'w') as file:
        file.write(r"\begin{table}[h!]" + "\n")
        file.write(r"    \centering" + "\n")
        file.write(r"    \begin{tabular}{|l|c|}" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \textbf{Metric} & \textbf{Value} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Audio Quality}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Total audio files & {total_files} \\\\" + "\n")
        file.write(f"        Mean duration & {mean_duration:.2f} seconds \\\\" + "\n")
        file.write(f"        Max duration & {max_duration:.2f} seconds \\\\" + "\n")
        file.write(f"        Min duration & {min_duration:.2f} seconds \\\\" + "\n")
        file.write(f"        Mean sample rate & {mean_sample_rate:.2f} Hz \\\\" + "\n")
        file.write(f"        Sample rate standard deviation & {std_sample_rate:.2f} Hz \\\\" + "\n")
        file.write(f"        Mean volume & {mean_volume:.2f} dB \\\\" + "\n")
        file.write(f"        Volume standard deviation & {std_volume:.2f} dB \\\\" + "\n")
        # file.write(f"        Mean SNR & {mean_snr:.2f} dB \\\\" + "\n")
        # file.write(f"        SNR standard deviation & {std_snr:.2f} dB \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Speech Characteristics}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Total pauses & {sum(total_pauses_list)} \\\\" + "\n")
        file.write(f"        Mean pauses per audio file: {avg_pauses_per_audio:.2f} \\\\" + "\n")
        file.write(f"        Mean pause duration & {mean_pause_duration:.2f} seconds \\\\" + "\n")
        file.write(f"        Pause duration standard deviation & {std_pause_duration:.2f} seconds \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"        \multicolumn{2}{|c|}{\textbf{Speaker Characteristics}} \\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(f"        Mean pitch & {mean_pitch:.2f} Hz \\\\" + "\n")
        file.write(f"        Pitch standard deviation & {std_pitch:.2f} Hz \\\\" + "\n")
        file.write(f"        Max pitch & {max_pitch_value:.2f} Hz \\\\" + "\n")
        file.write(f"        Min pitch & {min_pitch_value:.2f} Hz \\\\" + "\n")
        file.write(r"        \hline" + "\n")
        file.write(r"    \end{tabular}" + "\n")
        file.write(r"    \caption{Audio Analysis Metrics}" + "\n")
        file.write(r"    \label{tab:audio_analysis}" + "\n")
        file.write(r"\end{table}" + "\n")


    df_output = pd.DataFrame({
        'file_path': file_paths,
        'duration': durations,
        'sample_rate': sample_rates,
        'volume': volumes,
        # 'snr': snrs,
        'total_pauses': total_pauses_list,
        'avg_pauses_per_audio': avg_pauses_per_audio,
        'avg_pause_duration': avg_pause_durations,
        'mean_pitch': mean_pitches,
        'std_pitch': std_pitches,
        'max_pitch': max_pitches,
        'min_pitch': min_pitches
    })


    print(f"Statistics saved to {output_text_file}")
    print(f"LaTeX table saved to {output_latex_file}")
    df_output.to_csv(output_csv_file, index=False)
    print(f"Analysis results saved to {output_text_file}")

    plot_metrics(df_output, output_folder, database_name)


def main():
    base_directory = os.getcwd()
    files = [os.path.join(root, file) for root, _, files in os.walk(base_directory) for file in files if file.endswith('_transcriptions.csv')]

    for file in tqdm(files, desc="Processing transcription files"):
        if "Europarl-TS" in file:
            # top_db = 40
            # min_pause_duration = 0.1
            continue
        
        elif "M-AILABS" in file:
            # top_db = 30
            # min_pause_duration = 0.05
            continue
        
        elif "MintzAI-ST" in file:
            # top_db = 21
            # min_pause_duration = 0.13
            continue

        elif "Common_Voice_v9.0" in file or "Common_Voice_v12.0" in file or "Common_Voice_v15.0" in file:
            # top_db = 21
            # min_pause_duration = 0.13
            continue

        elif "ASR-SpCSC" in file:
            # top_db = 20
            # min_pause_duration = 0.10
            continue

        elif "King-ASR-L-202" in file:
            # top_db = 20
            # min_pause_duration = 0.1
            continue
        
        elif "ALBAYZIN2016_ASR" in file:
            # top_db = 11
            # min_pause_duration = 0.15
            continue

        elif "TTS_DB" in file:
            top_db = 11
            min_pause_duration = 0.15
            # continue

        elif "Parlamento_EJ" in file:
            # top_db = 26
            # min_pause_duration = 0.15
            continue

        elif "120h_Spanish_Speech" in file:
            # top_db = 20
            # min_pause_duration = 0.05
            continue
        
        elif "OpenSLR" in file:
            # top_db = 21
            # min_pause_duration = 0.15
            continue
        
        else:
            print(f"No matching database for file: {file}")
            continue

        folder_path = file.split("/")[:10]
        folder_path = os.path.join("/", *folder_path)
        database_name = file.split("/")[9]
        
        output_folder = os.path.join(folder_path, 'acoustics')
        os.makedirs(output_folder, exist_ok=True)

        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_csv(file, on_bad_lines='skip')


        print(f"Processing: {database_name}")
        print(f"Using top_db: {top_db}")
        print(f"Using min pause dur: {min_pause_duration}")
        print(file)
        print(len(df))
        # exit()
        analyze_audio_files(df, output_folder, database_name, top_db, min_pause_duration)


if __name__ == "__main__":
    main()
