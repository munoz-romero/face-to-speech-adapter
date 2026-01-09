#!/usr/bin/env python3
"""
Prepara el archivo de metadata de LRS3 para entrenamiento de StyleTTS2.

Convierte el formato:
    id,transcription
    GYzLcpN8dso_50001,I BOUGHT THE CURRICULUM AND I LEARNED IT

Al formato StyleTTS2:
    /path/to/audio.wav|phonemes|speaker_id

Uso:
    python prepare_lrs3_metadata.py \
        --input /home/voces/datasets/lrs3/trainval_processed/transcriptions.csv \
        --output /home/voces/datasets/lrs3/trainval_processed/metadata_train.csv \
        --audio-dir /home/voces/datasets/lrs3/trainval_processed/audio \
        --language en-us \
        --num-workers 8
"""

import argparse
import csv
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Suprimir warnings de phonemizer
warnings.filterwarnings('ignore')


def setup_phonemizer(language: str):
    """Configura el backend de phonemizer."""
    import phonemizer
    return phonemizer.backend.EspeakBackend(
        language=language,
        preserve_punctuation=True,
        with_stress=True
    )


def phonemize_text(text: str, phonemizer_backend) -> str:
    """
    Fonemiza un texto usando espeak.

    Args:
        text: Texto en inglés (mayúsculas o minúsculas)
        phonemizer_backend: Backend de phonemizer

    Returns:
        Texto fonemizado
    """
    # Normalizar: minúsculas y limpiar
    text = text.strip().lower()

    # Añadir puntuación si no tiene
    if text and text[-1].isalnum():
        text = text + "."

    try:
        # Fonemizar
        phonemes = phonemizer_backend.phonemize([text])
        if phonemes:
            return phonemes[0].strip()
    except Exception as e:
        print(f"Error fonemizando '{text[:50]}...': {e}")

    return ""


def extract_speaker_id(sample_id: str) -> str:
    """
    Extrae el ID del speaker del ID de muestra.

    Formato LRS3: VIDEOID_CLIPID (ej: GYzLcpN8dso_50001)
    El VIDEOID corresponde típicamente al mismo speaker.
    """
    parts = sample_id.split('_')
    if len(parts) >= 2:
        return parts[0]  # VIDEOID como speaker_id
    return sample_id


def process_batch(batch, audio_dir, language):
    """Procesa un batch de muestras (para paralelización)."""
    # Crear phonemizer en cada proceso
    phonemizer_backend = setup_phonemizer(language)

    results = []
    for sample_id, transcription in batch:
        # Construir path de audio
        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")

        # Verificar que existe
        if not os.path.exists(audio_path):
            continue

        # Fonemizar
        phonemes = phonemize_text(transcription, phonemizer_backend)
        if not phonemes:
            continue

        # Extraer speaker_id
        speaker_id = extract_speaker_id(sample_id)

        results.append((audio_path, phonemes, speaker_id))

    return results


def main():
    parser = argparse.ArgumentParser(description='Prepara metadata LRS3 para StyleTTS2')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Archivo CSV de entrada (transcriptions.csv)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Archivo CSV de salida (metadata_train.csv)')
    parser.add_argument('--audio-dir', '-a', type=str, required=True,
                        help='Directorio con archivos de audio .wav')
    parser.add_argument('--language', '-l', type=str, default='en-us',
                        help='Idioma para fonemización (default: en-us)')
    parser.add_argument('--num-workers', '-n', type=int, default=4,
                        help='Número de workers para paralelización')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='Tamaño de batch para procesamiento')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limitar número de muestras (para debug)')
    parser.add_argument('--val-split', type=float, default=0.0,
                        help='Porcentaje para validación (0.0-1.0). Si > 0, genera archivo _val.csv')

    args = parser.parse_args()

    # Verificar archivos
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No se encuentra: {args.input}")

    if not os.path.exists(args.audio_dir):
        raise FileNotFoundError(f"No se encuentra directorio: {args.audio_dir}")

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Leer archivo de entrada
    print(f"Leyendo {args.input}...")
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['id']
            transcription = row['transcription']
            samples.append((sample_id, transcription))

    print(f"Leídas {len(samples)} muestras")

    # Limitar si se especifica
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limitado a {len(samples)} muestras")

    # Dividir en batches
    batches = []
    for i in range(0, len(samples), args.batch_size):
        batches.append(samples[i:i + args.batch_size])

    print(f"Procesando {len(batches)} batches con {args.num_workers} workers...")

    # Procesar en paralelo
    all_results = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_batch, batch, args.audio_dir, args.language): i
            for i, batch in enumerate(batches)
        }

        with tqdm(total=len(batches), desc="Fonemizando") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error en batch: {e}")
                pbar.update(1)

    print(f"Procesadas {len(all_results)} muestras válidas")

    # Obtener speaker_ids únicos para asignar IDs numéricos
    unique_speakers = sorted(set(r[2] for r in all_results))
    speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
    print(f"Speakers únicos: {len(unique_speakers)}")

    # Split train/val si se especifica
    if args.val_split > 0:
        import random
        random.seed(42)
        random.shuffle(all_results)

        val_size = int(len(all_results) * args.val_split)
        val_results = all_results[:val_size]
        train_results = all_results[val_size:]

        # Escribir archivo de validación
        val_output = args.output.replace('.csv', '_val.csv')
        with open(val_output, 'w', encoding='utf-8') as f:
            for audio_path, phonemes, speaker_id in val_results:
                numeric_id = speaker_to_id[speaker_id]
                f.write(f"{audio_path}|{phonemes}|{numeric_id}\n")
        print(f"Guardado {len(val_results)} muestras de validación en {val_output}")

        all_results = train_results

    # Escribir archivo de salida
    with open(args.output, 'w', encoding='utf-8') as f:
        for audio_path, phonemes, speaker_id in all_results:
            numeric_id = speaker_to_id[speaker_id]
            f.write(f"{audio_path}|{phonemes}|{numeric_id}\n")

    print(f"Guardado {len(all_results)} muestras en {args.output}")

    # Guardar mapeo de speakers
    speaker_map_path = args.output.replace('.csv', '_speakers.txt')
    with open(speaker_map_path, 'w', encoding='utf-8') as f:
        for spk, idx in sorted(speaker_to_id.items(), key=lambda x: x[1]):
            f.write(f"{idx}|{spk}\n")
    print(f"Mapeo de speakers guardado en {speaker_map_path}")

    # Estadísticas finales
    print("\n=== Estadísticas ===")
    print(f"Total muestras: {len(all_results)}")
    print(f"Total speakers: {len(unique_speakers)}")

    # Mostrar ejemplo
    if all_results:
        print("\n=== Ejemplo de salida ===")
        for i, (audio_path, phonemes, speaker_id) in enumerate(all_results[:3]):
            print(f"{audio_path}|{phonemes}|{speaker_to_id[speaker_id]}")


if __name__ == '__main__':
    main()
