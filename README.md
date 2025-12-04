# Evaluate_WER_ES_EU_databases

## Overview

This repository contains the companion code for a Master’s thesis on evaluating Automatic Speech Recognition (ASR) datasets for Spanish and Basque.

The main goal is to analyze text and audio before training in order to assess how “good” a dataset is for training and evaluating STT/ASR models. Instead of only looking at final WER numbers, this project characterizes:

- The linguistic properties of the transcriptions  
- The acoustic properties of the corresponding audio  
- How these factors relate to ASR performance (WER, error types, hallucinations, etc.)

Although the case study focuses on Basque (with Spanish support), the tooling is designed to be reusable for other languages and corpora.

---

## What this repo does

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
- `images/STT_Models_Error_Histograms_es.png` → WER distribution histograms
- `images/WER_vs_Duration_Three_Models_Heatmap.png` → WER as a function of audio duration
- `images/WER_vs_Words_in_Reference_DeepSpeech_Whisper_no_LM.png` → WER vs. number of words in the reference
- `images/hallucination_percent_comparison.png` → Hallucination-heavy output percentages (LM vs no-LM)

#### Corresponding tables (CSV/LaTeX/Markdown):
- WER per model and dataset (e.g., `wer_basque_results.csv`)  
- Error tables for:
  - Proper nouns and short words  
  - Long-form substitutions  
  - Hallucination cases

---

### 2. Perform linguistic analysis of transcriptions

Scripts and notebooks compute linguistic metrics over each dataset’s reference transcripts. These quantify textual difficulty and diversity and explain why some corpora are harder for ASR than others.

#### Syntactic complexity
- Sentence length: words per sentence  
- Parse tree depth: average and standard deviation  
- Clauses per sentence: average and standard deviation

**Key figure:**  
`images/syntactical_metrics_eu.png` → Words per sentence, average parse tree depth, and clauses per sentence per dataset.

#### Lexical richness and diversity
- Vocabulary density  
- MTLD (Measure of Textual Lexical Diversity)  
- RTTR (Root Type-Token Ratio)

**Key table & figure:**
- `lexical_diversity_eu.csv` → Vocabulary density per dataset  
- `images/RTTR_metrics_eu.png` → MTLD and RTTR values, highlighting diversity vs repetition

#### Part-of-speech (PoS) distributions
- Proportions of nouns, verbs, auxiliaries, adjectives, adverbs, etc.  
- Helps identify “narrative”, “informative”, and “technical” styles.

**Key figure:**  
`images/morphology_metrics_eu.png` → PoS distribution per dataset (Basque)

---

### 3. Perform acoustic analysis of the audio

The repo computes acoustic descriptors for each file and aggregates them per dataset to explain variability from recording conditions and speaking style.

#### Temporal structure
- File duration statistics (mean, std)  
- Pause count and duration (number, mean, variability)

**Key figures:**  
- `images/time_duration_metrics_eu.png` → Average audio and pause duration (+ std)  
- `images/pauses_metrics_eu.png` → Average number of pauses and variations  

#### Loudness / volume
- Mean volume per file/dataset (dBFS)  
- Standard deviation of volume  

**Key figure:**  
`images/volume_metrics.png` → Volume statistics per dataset  

#### Pitch & sample rate
- Check sample rates (≥ 16 kHz, no upsampling)  
- Pitch statistics: mean pitch, pitch std (vocal variability)  

**Key figure:**  
`images/pitch_metrics_eu.png` → Pitch metrics per dataset  

These descriptors are exported as CSV tables for further analysis.

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
