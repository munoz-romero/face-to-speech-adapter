#!/usr/bin/env python3
"""
Script para procesar el dataset LRS3.
Extrae audio (WAV), un fotograma aleatorio de la cara, y transcripciones de los vídeos.
"""

import os
import sys
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm
import random
import csv
import argparse


def extract_audio(video_path, output_audio_path):
    """
    Extrae el audio de un vídeo MP4 y lo guarda como WAV.

    Args:
        video_path: Ruta al archivo de vídeo MP4
        output_audio_path: Ruta donde guardar el archivo WAV
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Codec de audio WAV
            '-ar', '24000',  # Sample rate 16kHz
            '-ac', '1',  # Mono
            '-y',  # Sobrescribir si existe
            '-loglevel', 'error',  # Solo mostrar errores
            str(output_audio_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extrayendo audio de {video_path}: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado extrayendo audio de {video_path}: {e}")
        return False


def extract_random_frame(video_path, output_image_path):
    """
    Extrae un fotograma aleatorio de un vídeo y lo guarda como imagen.

    Args:
        video_path: Ruta al archivo de vídeo MP4
        output_image_path: Ruta donde guardar la imagen
    """
    try:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error abriendo vídeo: {video_path}")
            return False

        # Obtener el número total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"No se pudieron obtener frames de: {video_path}")
            cap.release()
            return False

        # Seleccionar un frame aleatorio
        random_frame = random.randint(0, total_frames - 1)

        # Ir al frame aleatorio
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        # Leer el frame
        ret, frame = cap.read()

        if ret:
            # Guardar el frame como imagen
            cv2.imwrite(str(output_image_path), frame)
            cap.release()
            return True
        else:
            print(f"Error leyendo frame de: {video_path}")
            cap.release()
            return False

    except Exception as e:
        print(f"Error extrayendo frame de {video_path}: {e}")
        return False


def extract_transcription(txt_path):
    """
    Extrae la transcripción de un archivo de texto LRS3.

    Args:
        txt_path: Ruta al archivo .txt

    Returns:
        String con la transcripción o None si hay error
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

            # El formato es "Text: TRANSCRIPTION"
            if first_line.startswith("Text:"):
                transcription = first_line.replace("Text:", "").strip()
                return transcription
            else:
                print(f"Formato inesperado en {txt_path}: {first_line}")
                return None

    except Exception as e:
        print(f"Error leyendo transcripción de {txt_path}: {e}")
        return None


def process_lrs3_directory(input_dir, output_dir, csv_output_path):
    """
    Procesa todos los vídeos MP4 en un directorio LRS3.

    Args:
        input_dir: Directorio raíz que contiene subdirectorios con archivos MP4
        output_dir: Directorio donde guardar los archivos procesados
        csv_output_path: Ruta del archivo CSV con las transcripciones
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Crear directorio de salida si no existe
    output_path.mkdir(parents=True, exist_ok=True)

    # Crear subdirectorios para audio e imágenes
    audio_dir = output_path / "audio"
    images_dir = output_path / "images"
    audio_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    # Buscar todos los archivos MP4
    mp4_files = list(input_path.rglob("*.mp4"))

    print(f"Encontrados {len(mp4_files)} archivos MP4 en {input_dir}")

    # Lista para almacenar las transcripciones
    transcriptions = []

    # Procesar cada vídeo
    successful = 0
    failed = 0

    for video_path in tqdm(mp4_files, desc="Procesando vídeos"):
        # Obtener el ID del vídeo (nombre sin extensión)
        # El ID incluye el subdirectorio y el nombre del archivo
        relative_path = video_path.relative_to(input_path)
        video_id = f"{relative_path.parent.name}_{relative_path.stem}"

        # Rutas de salida
        audio_output = audio_dir / f"{video_id}.wav"
        image_output = images_dir / f"{video_id}.jpg"

        # Ruta del archivo de transcripción
        txt_path = video_path.with_suffix('.txt')

        # Verificar que existe el archivo de texto
        if not txt_path.exists():
            print(f"\nAdvertencia: No se encontró archivo de texto para {video_path}")
            failed += 1
            continue

        # Extraer transcripción
        transcription = extract_transcription(txt_path)
        if transcription is None:
            failed += 1
            continue

        # Extraer audio
        audio_success = extract_audio(video_path, audio_output)

        # Extraer frame aleatorio
        frame_success = extract_random_frame(video_path, image_output)

        # Si ambos fueron exitosos, agregar a la lista de transcripciones
        if audio_success and frame_success:
            transcriptions.append({
                'id': video_id,
                'transcription': transcription
            })
            successful += 1
        else:
            failed += 1

    # Guardar transcripciones en CSV
    print(f"\nGuardando transcripciones en {csv_output_path}")
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in transcriptions:
            writer.writerow(item)

    print(f"\n{'='*60}")
    print(f"Procesamiento completado:")
    print(f"  - Exitosos: {successful}")
    print(f"  - Fallidos: {failed}")
    print(f"  - Total: {len(mp4_files)}")
    print(f"\nArchivos guardados en:")
    print(f"  - Audio: {audio_dir}")
    print(f"  - Imágenes: {images_dir}")
    print(f"  - Transcripciones: {csv_output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Procesar dataset LRS3: extraer audio, frames y transcripciones'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directorio de entrada con los datos LRS3 (ej: /home/voces/datasets/lrs3/pretrain)'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directorio de salida para los archivos procesados'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Ruta del archivo CSV de salida (por defecto: output_dir/transcriptions.csv)'
    )

    args = parser.parse_args()

    # Si no se especifica CSV, usar el directorio de salida
    csv_path = args.csv
    if csv_path is None:
        csv_path = os.path.join(args.output_dir, 'transcriptions.csv')

    # Verificar que el directorio de entrada existe
    if not os.path.exists(args.input_dir):
        print(f"Error: El directorio de entrada no existe: {args.input_dir}")
        sys.exit(1)

    # Procesar
    process_lrs3_directory(args.input_dir, args.output_dir, csv_path)


if __name__ == "__main__":
    main()
