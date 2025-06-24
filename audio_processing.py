import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment, silence
import os

def load_and_process_audio(input_path, target_sr=16000):
    """Carga audio, convierte a mono, resamplea y normaliza (peak)."""
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)  # Carga mono y resamplea
    y = y / np.max(np.abs(y))  # Normalización peak [-1, 1]
    return y, target_sr

def save_audio(y, sr, output_path):
    """Guarda audio como WAV 16-bit PCM."""
    y_int16 = (y * 32767).astype(np.int16)  # Escala a int16
    sf.write(output_path, y_int16, sr, subtype='PCM_16')

def normalize_with_pydub(input_path, output_path):
    """Normaliza volumen a aprox -20 dBFS usando pydub."""
    audio = AudioSegment.from_file(input_path)
    change_in_dBFS = -20.0 - audio.dBFS  # Calcula ganancia necesaria
    normalized_audio = audio.apply_gain(change_in_dBFS)  # Aplica ganancia
    normalized_audio.export(output_path, format="wav")  # Guarda

def trim_silence(input_path, output_path, silence_thresh=-40, min_silence_len=300):
    """Recorta silencios al inicio/fin usando pydub."""
    audio = AudioSegment.from_file(input_path)
    # Detecta rangos no silenciosos
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if nonsilent_ranges:
        # Recorta al primer y último segmento con voz
        start_trim = nonsilent_ranges[0][0]
        end_trim = nonsilent_ranges[-1][1]
        trimmed_audio = audio[start_trim:end_trim]
        trimmed_audio.export(output_path, format="wav")  # Guarda recortado
        print(f"Guardado audio recortado: {output_path}")
    else:
        # Si no hay voz, guarda el original
        print(f"No se detectó voz en {input_path}, se guarda el audio original.")
        audio.export(output_path, format="wav")

def process_directory(input_dir, output_dir):
    """Procesa todos los .wav de un directorio: normaliza y recorta silencios."""
    os.makedirs(output_dir, exist_ok=True)  # Crea carpeta destino si no existe
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".wav"):
            input_path = os.path.join(input_dir, file_name)
            temp_path = os.path.join(output_dir, f"temp_{file_name}")
            output_path = os.path.join(output_dir, file_name)

            # Cargar, mono, resamplear y peak-normalizar
            y, sr = load_and_process_audio(input_path)
            sf.write(temp_path, y, sr)

            # Normalización a -20 dBFS
            normalize_with_pydub(temp_path, temp_path)

            # Recorte de silencios
            trim_silence(temp_path, output_path)

            # Elimina temporal
            os.remove(temp_path)
            print(f"Procesado: {file_name}")

def main():
    """Main: procesa audios de ./audios y guarda en ./audios_cleaned."""
    input_dir = "./audios"
    output_dir = "./audios_cleaned"
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()