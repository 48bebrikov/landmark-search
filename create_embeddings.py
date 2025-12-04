import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import os
import warnings


MODEL_ID = "google/siglip2-so400m-patch14-384"
SOURCE_CSV = "dataset_indexed.csv" 
IMAGES_DIR = "images"
BATCH_SIZE = 256

OUTPUT_EMBEDDINGS = "embeddings.npy"
OUTPUT_CSV = "dataset_indexed.csv"

warnings.filterwarnings("ignore")

def main():
    # 1. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 2. Load Model & Processor
    print(f"Loading model: {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # 3. Prepare Data
    if not os.path.exists(SOURCE_CSV):
        print(f"Error: Source CSV '{SOURCE_CSV}' not found!")
        return

    df = pd.read_csv(SOURCE_CSV)
    print(f"Found {len(df)} rows in CSV.")

    image_paths = []
    valid_indices = []

    print("Validating images...")
    for idx, row in df.iterrows():
        fname = row['filename']
        path = os.path.join(IMAGES_DIR, fname)
        
        if os.path.exists(path):
            image_paths.append(path)
            valid_indices.append(idx)
    
    if not image_paths:
        print("No valid images found! Check IMAGES_DIR path.")
        return

    df_clean = df.loc[valid_indices].reset_index(drop=True)
    print(f"Ready to vectorize: {len(df_clean)} images.")

    # 4. Inference Loop
    all_embeddings = []
    
    print("Starting inference...")
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]

        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception as e:
            print(f"Error loading batch {i}: {e}")
            continue

        with torch.no_grad():
            # Preprocessing
            inputs = processor(images=images, return_tensors="pt").to(device)
            
            # Model Inference
            features = model.get_image_features(**inputs)

            # Normalization (Critical for Cosine Similarity!)
            # x / ||x||
            features = features / features.norm(p=2, dim=-1, keepdim=True)

            # Move to CPU and list
            all_embeddings.append(features.cpu().numpy())

    # 5. Save Results
    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        
        print(f"Saving embeddings shape: {final_embeddings.shape}...")
        np.save(OUTPUT_EMBEDDINGS, final_embeddings)
        
        print(f"Saving metadata to {OUTPUT_CSV}...")
        df_clean.to_csv(OUTPUT_CSV, index=False)
        
        print("Done! You can now run app.py or docker compose up.")
    else:
        print("Failed to generate embeddings.")

if __name__ == "__main__":
    main()

