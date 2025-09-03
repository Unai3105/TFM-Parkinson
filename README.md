# TFM-Parkinson

## Repository Overview
The **TFM-Parkinson** repository contains two distinct systems:

1. **DDoS Detection Research System:** A theoretical system documented in academic literature.  
2. **Neurovoz v3 Audio Processing System:** A fully implemented audio processing pipeline for Parkinson's disease detection via voice analysis.

For detailed documentation, see the corresponding sections: [DDoS Detection Research System](#) and [Neurovoz v3 Audio Processing System](#).

---

## Repository Architecture Overview
The repository follows a **dual-system architecture**, with two independent systems serving different research purposes.

<!-- Insert image here: Repository Architecture -->
![Execution Flow](https://github.com/user-attachments/assets/8e25a4cf-b5b4-4fc4-bbd8-c5eaaf9e8418)

---

## System Components Overview
The primary implementation is the **Neurovoz v3 audio processing pipeline**, with clearly defined components and data flow patterns.

### Core Components

| Component | Input Interface | Output Interface | Key Dependencies |
|-----------|----------------|----------------|-----------------|
| `audio_processing.py` | `./audios/*.wav` | `./audios_cleaned/*.wav` | librosa, pydub, soundfile |
| `audio_features_extracting.py` | `./audios_cleaned/*.wav` | `audio_features_new.csv` | parselmouth, librosa, pandas |
| `neurovoz_v3_metadatos.ipynb` | `./audios_cleaned/*.wav` | `neurovoz_v3_metadatos.csv`, `*.npy` | librosa, numpy |
| `neurovoz_v3_audios_new.ipynb` | `features_all.csv`, `metadata.csv` | Trained models, metrics | transformers, torch, sklearn |

---

## Data Processing Pipeline
The system uses a **file-based architecture**, where components communicate through structured files rather than direct API calls, allowing robust batch processing.

<!-- Insert image here: Data Processing Pipeline -->
![Data Processing Pipeline](https://github.com/user-attachments/assets/bf2b65a9-0b66-472c-af2d-c22116466ed5)

---

## Technical Implementation Details

### Audio Processing Configuration
- **Sample Rate:** 16kHz target resampling via `librosa.load(target_sr=16000)`  
- **Normalization:** -20dBFS volume normalization using `pydub.AudioSegment.apply_gain()`  
- **Silence Detection:** VAD with -40dB threshold and 300ms minimum silence duration  
- **Format:** 16-bit PCM WAV output via `soundfile.write(subtype='PCM_16')`  

### Feature Extraction Architecture
- **Vocal Features:** F0 statistics, jitter/shimmer analysis via `parselmouth.praat.call()`  
- **Spectral Features:** 13 MFCC coefficients with mean/std statistics via `librosa.feature.mfcc()`  
- **Cepstral Analysis:** Custom CPP implementation in `cepstral_peak_prominence()`  
- **Voice Quality:** HNR analysis through `snd.to_harmonicity_cc()`

### Machine Learning Pipeline
- **Model:** Wav2Vec2 via `transformers.Wav2Vec2Model`  
- **Framework:** PyTorch backend with scikit-learn preprocessing  
- **Features:** Combined acoustic features and raw audio embeddings  
- **Evaluation:** Cross-validation with ROC analysis and confusion matrices  

---

## Sources
- `TFM - Unai Roa.pdf`  
- `audio_processing.py`  
- `audio_features_extracting.py`  
- `neurovoz_v3_audios.ipynb`  
- `neurovoz_v3_audios_new.ipynb`
