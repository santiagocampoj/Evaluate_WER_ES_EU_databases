# Evaluate_WER_ES_EU_databases

## Overview

This repository contains the companion code for a Master’s thesis on evaluating Automatic Speech Recognition (ASR) datasets for Spanish and Basque.

The main goal is to analyze text and audio before training in order to assess how “good” a dataset is for training and evaluating STT/ASR models. Instead of only looking at final WER numbers, this project characterizes:

- The linguistic properties of the transcriptions  
- The acoustic properties of the corresponding audio  
- How these factors relate to ASR performance (WER, error types, hallucinations, etc.)

Although the case study focuses on Basque (with Spanish support), the toolkit is designed to be reusable for other languages and corpora.

---

## What this repo does (taking the EU analysis example)

### 1. Compute WER and detailed error breakdowns

The code computes Word Error Rate (WER) and error statistics for several Basque STT models:

- Basque DeepSpeech STT v0.1.7  
- Basque Whisper Medium  
  - with Language Model (LM)  
  - without Language Model (no-LM)  

For each model–dataset pair, the pipeline extracts:

- Global WER  
- Error type breakdown:  
  - Substitutions  
  - Deletions  
  - Insertions  
  - Hallucinations (over-generation of words)

#### Key outputs / figures (stored in `images/`):
- `images/STT_Models_Error_eu.png` → WER per model and dataset (bar plot)
- <img width="400" height="1000" alt="STT_Models_Error_eu" src="https://github.com/user-attachments/assets/297060b8-68c9-4ca6-8c60-e1bcee2ffa68" />

- `images/STT_Models_Error_Histograms_es.png` → WER distribution histograms
<img width="700" height="500" alt="STT_Models_Error_Histograms_es" src="https://github.com/user-attachments/assets/e46badb8-1df6-476c-8a1f-e3da6b91af49" />
  
- `images/WER_vs_Duration_Three_Models_Heatmap.png` → WER as a function of audio duration
<img width="1800" height="600" alt="WER_vs_Duration_Three_Models_Heatmap" src="https://github.com/user-attachments/assets/502296f1-3d78-4459-937f-e3272955a6ad" />

  
- `images/WER_vs_Words_in_Reference_DeepSpeech_Whisper_no_LM.png` → WER vs. number of words in the reference
<img width="500" height="700" alt="WER_vs_Words_in_Reference_DeepSpeech_Whisper_no_LM" src="https://github.com/user-attachments/assets/b15d4eef-952a-4eed-981d-02185b6637ed" />

  
- `images/hallucination_percent_comparison.png` → Hallucination-heavy output percentages (LM vs no-LM)
  <img width="500" height="700" alt="hallucination_percent_comparison" src="https://github.com/user-attachments/assets/c85bca2f-3379-45b3-a76f-e8e64d607b35" />

  

#### Corresponding tables):
- WER per model and dataset (e.g., `wer_basque_results.csv`)  
- Error tables for:
  - Proper nouns and short words  
  - Long-form substitutions  
  - Hallucination cases

---

### 1. Linguistic analysis (text side)

We characterize the transcriptions of each dataset before training:

- **Syntactic complexity**  
  Plots like `images/syntactical_metrics_eu.png` show:
  - words per sentence  
  - parse-tree depth (how nested sentences are)  
  - clauses per sentence and their variability
    
<img width="700" height="700" alt="syntactical_metrics_eu" src="https://github.com/user-attachments/assets/76175345-6953-48e3-99f9-795f7bc57219" />


- **Lexical diversity**  
  Tables and plots such as:
  - `Vocabulary Density` table  
  - `images/RTTR_metrics_eu.png`  
  report:
  - vocabulary density  
  - MTLD (lexical richness)  
  - RTTR (type–token ratio with length normalisation)
 
  <img width="600" height="500" alt="RTTR_metrics_eu_DeepSpeech" src="https://github.com/user-attachments/assets/b9e93e4b-d40e-4564-a153-d0250d6017e6" />


- **Part-of-Speech profiles**  
  `images/morphology_metrics_eu.png` shows PoS distributions  
  (nouns, verbs, auxiliaries…) to see whether a dataset is full of entities, actions, or more “function words”.

  <img width="700" height="900" alt="morphology_metrics_eu" src="https://github.com/user-attachments/assets/b9742e1f-e06d-44d9-96bc-5ce5b6a284e4" />


These metrics let us see which datasets are linguistically simple (short, repetitive, limited vocabulary) vs. complex (long, nested sentences with rich vocabulary and many unique words).

---

### 2. Audio analysis (signal side)

We also analyse the **audio recordings themselves** to understand how hard they are for an ASR model:

- **Duration & pauses**  
  - `images/time_duration_metrics_eu.png` (average file duration + std)
  - `images/pauses_metrics_eu.png` (number and length of pauses)  
  We study:
  - how long typical utterances are  
  - how variable their duration is  
  - how often speakers pause and for how long

 <img width="700" height="900" alt="time_duration_metrics_eu" src="https://github.com/user-attachments/assets/4ecc6224-8a37-4210-9452-1dca24f54324" />

 <img width="700" height="900" alt="pauses_metrics_eu" src="https://github.com/user-attachments/assets/c80feaec-779c-492f-beb0-6d40070e1f94" />


    

- **Loudness & dynamics**  
  - `images/volume_metrics.png`  
  shows mean volume (in dB) and its variability, to detect datasets with very uneven levels that can confuse models.

<img width="700" height="900" alt="volume_metrics" src="https://github.com/user-attachments/assets/95af5fc9-1610-4249-a7c4-aab6510d4489" />


- **Pitch & sample rate**  
  - `images/pitch_metrics_eu.png`  
  summarizes mean pitch and pitch variability per corpus and checks whether sample rates are already high enough for ASR (≥16 kHz).

<img width="700" height="900" alt="pitch_metrics_eu" src="https://github.com/user-attachments/assets/8b4a9536-524f-4723-ad43-169569f532d3" />


This side of the analysis tells us which datasets have **clean, consistent recordings** vs. those with **noisy, variable, or very long/uneven audio**, which usually make recognition harder.

---

## Datasets

Experiments were run on multiple Basque (and optionally Spanish) datasets:

- Common Voice v9  
- Common Voice v12  
- Common Voice v15  
- Banco de Voces  
- ADITU  
- TTS DB  
- OpenSLR (Basque subset)  
- MintzAI-ST  
- Parlamento_EJ  

Each dataset is characterized:

- **Linguistically:** Complexity, lexical diversity, PoS distributions  
- **Acoustically:** Duration, pauses, volume, pitch, sample rates  
- **ASR-wise:** WER, error types, hallucinations, WER vs duration/length

---

## Example insights (Basque case study)

Some of the main findings enabled by this codebase:

- **Dataset complexity matters.**  
  - Common Voice and TTS DB: moderate complexity, stable acoustics → lower WER  
  - MintzAI-ST and Parlamento EJ: long, nested sentences, high diversity → higher WER  

- **Whisper + LM usually best (with trade-offs).**  
  - LM reduces WER vs. DeepSpeech and Whisper no-LM  
  - LM increases insertions/hallucinations on complex data  

- **Short / long utterances are challenging.**  
  - Very short (<5 words) and very long (>40 words) utterances yield higher WER  
  - Basque morphology, proper names, and foreign words increase errors

---

## How to use this repo

*(Working on it)*

### 1. Prepare your datasets
Organize folder structure as:

data/<dataset_name>/audio
data/<dataset_name>/text


### 2. Run preprocessing & feature extraction
Use scripts in:
scripts/ or notebooks/

Examples:
- `linguistic_analysis_*.py`
- `acoustic_analysis_*.py` 
- `wer_analysis_*.py`

### 3. Generate figures and tables
Outputs include:  
- `.csv` tables for metrics  
- Figures in `images/` (as above)  

### 4. Extend to new corpora or languages
Provide new audio–transcript pairs and rerun the same pipelines to obtain all linguistic, acoustic, and WER metrics.

---

## License
This project is released under the GNU License. See `LICENSE` for details.

## Contact
For questions or collaboration: santiagocampojurado@gmail.com
