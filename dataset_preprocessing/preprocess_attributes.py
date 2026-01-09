import os
import sys
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

def main():
    parser = argparse.ArgumentParser(description="Extract Gender and Age attributes from LRS3 dataset using InsightFace")
    parser.add_argument('--data_path', type=str, required=True, help='Path to LRS3 dataset root (containing metadata_*.csv)')
    parser.add_argument('--output_file', type=str, default='face_attributes.json', help='Output JSON file path')
    args = parser.parse_args()

    # Initialize InsightFace
    print("Initializing InsightFace...")
    # Use CUDA if available, else CPU
    providers = ['CUDAExecutionProvider'] if insightface.model_zoo.model_zoo.onnxruntime.get_device() == 'GPU' else ['CPUExecutionProvider']
    # Note: onnxruntime check might be different depending on version, let's trust the environment or try-catch
    # Actually, let's just try to use CUDA, if it fails, it usually falls back or we can force it if we knew.
    # Given the previous context, we used:
    # providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    # But here we don't import torch necessarily. Let's import torch to check cuda.
    import torch
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))

    attributes = {}

    # Trainval_processed
    # metadata_files = ['metadata_train_validated.csv', 'metadata_train_val_validated.csv', 'metadata_test_validated.csv']
    # Pretrain_processed
    metadata_files = ['metadata_train.csv', 'metadata_val.csv']


    for meta_file in metadata_files:
        full_path = os.path.join(args.data_path, meta_file)
        if not os.path.exists(full_path):
            print(f"Warning: {full_path} not found. Skipping.")
            continue

        print(f"Processing {meta_file}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines):
            parts = line.strip().split('|')
            if len(parts) < 3:
                continue

            audio_path = parts[0]
            # Construct image path as in the dataset class
            img_path = audio_path.replace('/audio/', '/images/').replace('.wav', '.jpg')

            # If path is relative, join with data_path?
            # In LRS3Dataset, it seems audio_path in metadata might be absolute or relative.
            # Usually metadata contains absolute paths or paths relative to some root.
            # Let's assume they are absolute or we need to check.
            # If they are not absolute, we might need to prepend args.data_path if they are relative to it.
            # But looking at LRS3Dataset:
            # img_path = audio_path.replace...
            # image = Image.open(img_path)
            # It opens it directly. So likely absolute.

            if not os.path.exists(img_path):
                # Try prepending data_path if not absolute
                if not os.path.isabs(img_path):
                    test_path = os.path.join(args.data_path, img_path)
                    if os.path.exists(test_path):
                        img_path = test_path
                    else:
                        # print(f"Image not found: {img_path}")
                        continue
                else:
                    # print(f"Image not found: {img_path}")
                    continue

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = app.get(img)

                if len(faces) > 0:
                    # Take the largest face or the first one
                    # Usually the one with highest detection score or largest area.
                    # InsightFace sorts by det score usually? Or we can sort by area.
                    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]

                    gender = int(face.gender) # 1 for Male, 0 for Female (usually)
                    age = int(face.age)

                    attributes[audio_path] = {
                        "gender": gender,
                        "age": age
                    }
                else:
                    # No face detected
                    pass

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Extracted attributes for {len(attributes)} samples.")

    output_path = os.path.join(args.data_path, args.output_file)
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(attributes, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
