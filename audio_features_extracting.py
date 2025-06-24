import parselmouth
import numpy as np
import librosa
import pandas as pd
import os
from scipy.signal import find_peaks

def cepstral_peak_prominence(signal, sr):
    """
    Calcula la prominencia del pico cepstral (CPP) en una señal de audio.
    """
    n = int(0.05 * sr)  # Primeros 50 ms de la señal
    x = signal[:n]

    spectrum = np.fft.fft(x)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real

    quefrency_range = cepstrum[10:200]  # rango típico para CPP

    peaks, _ = find_peaks(quefrency_range)
    if len(peaks) == 0:
        return np.nan

    peak_values = quefrency_range[peaks]
    max_peak = np.max(peak_values)
    mean_val = np.mean(quefrency_range)

    return max_peak - mean_val

def extract_parselmouth_features(audio_path):
    """
    Extrae características acústicas básicas usando Parselmouth (Pitch, jitter, shimmer, HNR).
    Devuelve un diccionario con las features.
    """
    # Carga audio
    snd = parselmouth.Sound(audio_path)
    # Duración en segundos
    duration_s = snd.get_total_duration()

    # Extrae pitch (F0)
    pitch = snd.to_pitch()

    # Media de F0 en Hz
    mean_F0_Hz = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")

    # Desviación estándar de F0
    stdev_F0_Hz = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz")

    # PointProcess para jitter y shimmer
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 600)

    # Jitter (inestabilidad frecuencia fundamental)
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer (variaciones de amplitud)
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    rShimmer_local_dB = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_Shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_Shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    sAPQ_Shimmer_dda = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # HNR (relación armónicos-ruido)
    harmonicity = snd.to_harmonicity_cc()
    mean_HNR_dB = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    # Carga audio para CPP y MFCC
    y, sr = librosa.load(audio_path, sr=None)
    CPP = cepstral_peak_prominence(y, sr)

    features = {
        "duration_s": duration_s,
        "mean_F0_Hz": mean_F0_Hz,
        "stdev_F0_Hz": stdev_F0_Hz,
        # Jitter
        "jitter_local": jitter_local,
        "rJitter_rap": jitter_rap,
        "rJitter_ppq5": jitter_ppq5,
        "rJitter_ddp": jitter_ddp,
        # Shimmer
        "shimmer_local": shimmer_local,
        "rShimmer_local_dB": rShimmer_local_dB,
        "APQ3_Shimmer": apq3_Shimmer,
        "APQ5_Shimmer": apq5_Shimmer,
        "sAPQ_Shimmer_dda": sAPQ_Shimmer_dda,
        # Ruido y calidad vocal
        "mean_HNR_dB": mean_HNR_dB,
        # Cepstrales
        "CPP": CPP,
    }
    return features

def extract_librosa_features(audio_path, n_mfcc=13):
    """
    Extrae MFCCs (media y desviación estándar) usando librosa.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    features = {}
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = mfccs_mean[i]
        features[f"mfcc_{i+1}_std"] = mfccs_std[i]
    return features

def extract_features_from_audio(audio_path):
    """
    Extrae y combina características acústicas y espectrales para un archivo de audio.
    """
    features_pm = extract_parselmouth_features(audio_path)
    features_mfcc = extract_librosa_features(audio_path)
    features = {**features_pm, **features_mfcc}
    features["AudioPath"] = audio_path
    return features

def save_features_to_csv(features_list, csv_path):
    """
    Guarda una lista de diccionarios con características en un CSV.
    """
    df = pd.DataFrame(features_list)
    df.to_csv(csv_path, index=False)
    print(f"Guardado CSV en: {csv_path}")

def main():
    """
    Extrae características de todos los audios en un directorio y guarda en CSV.
    """
    audio_dir = "./audios_cleaned"  # Directorio con archivos de audio
    if not os.path.exists(audio_dir):
        print(f"ERROR: Directorio no encontrado: {audio_dir}")
        return

    features_list = []
    # Procesar todos los archivos con extensión .wav en el directorio
    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(audio_dir, filename)
            print(f"Procesando: {audio_path}")
            features = extract_features_from_audio(audio_path)
            features_list.append(features)

    if features_list:
        save_features_to_csv(features_list, "audio_features_new.csv")
    else:
        print("No se encontraron archivos WAV en el directorio.")

if __name__ == "__main__":
    main()
