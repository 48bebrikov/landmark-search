import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
import gradio as gr
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_ID = "google/siglip2-so400m-patch14-384"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "dataset_indexed.csv"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
IMAGES_DIR = os.path.join(BASE_DIR, "images")

class SearchEngine:
    def __init__(self):
        print("Loading model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        # Optimization for CPU if needed, otherwise skip for GPU
        # if self.device == "cpu":
        #    print("Applying quantization for CPU optimization...")
        #    # Using dynamic quantization for Linear layers on CPU
        #    # Suppressing the specific deprecation warning for now as torchao requires extra install
        #    with warnings.catch_warnings():
        #        warnings.simplefilter("ignore")
        #        self.model = torch.quantization.quantize_dynamic(
        #            self.model, {torch.nn.Linear}, dtype=torch.qint8
        #        )
        
        print("Loading database...")
        if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(METADATA_FILE):
            raise FileNotFoundError("Files embeddings.npy or dataset_indexed.csv not found! Place them next to the script.")
            
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.df = pd.read_csv(METADATA_FILE)
        
        # Normalize vectors
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        
        print("System ready!")

    def _get_embedding(self, text=None, image=None):
        with torch.no_grad():
            if image:
                # Ensure RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                out = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(text=[text], return_tensors="pt", padding="max_length").to(self.device)
                out = self.model.get_text_features(**inputs)
            
            vec = out.cpu().numpy()
            # DEBUG: Print first 5 dims to check consistency with Colab
            if image:
                print(f"Img Vector [:5]: {vec[0, :5]}")
            
            return vec / np.linalg.norm(vec, axis=1, keepdims=True)

    def search(self, query_text=None, query_image=None, top_k=5):
        # 1. Get query vector
        q_vec = self._get_embedding(text=query_text, image=query_image)
        
        # 2. Fast search
        scores = np.dot(q_vec, self.embeddings.T).flatten()
        
        # DEBUG INFO
        print(f"Search Query: Text={bool(query_text)}, Image={bool(query_image)}")
        print(f"Top 3 Raw Scores: {np.sort(scores)[::-1][:3]}")

        # 3. Sort
        top_indices = np.argsort(scores)[::-1][:top_k*5] # Get more candidates to filter
        
        results = []
        seen_names = set()
        
        for idx in top_indices:
            row = self.df.iloc[idx]
            fname = row['filename']
            name = row['place_name']
            
            # Skip duplicates
            # if name in seen_names: continue
            # seen_names.add(name)
            
            # Check if image exists
            img_path = os.path.join(IMAGES_DIR, fname)
            if not os.path.exists(img_path):
                continue # Skip if image is missing
            
            results.append({
                "image": img_path,
                "caption": f"{name}\n({row['category']})\nScore: {scores[idx]:.2f}",
                "name": name,
                "category": row['category'],
                "score": scores[idx]
            })
            
            if len(results) >= top_k: break
            
        return results

    def classify(self, image):
        neighbors = self.search(query_image=image, top_k=10)
        
        name_votes = {}
        cat_votes = {}
        
        for n in neighbors:
            w = n['score']
            # Vote for name
            name_votes[n['name']] = name_votes.get(n['name'], 0) + w
            
            # Vote for category
            cats = str(n['category']).split(',')
            for c in cats:
                c = c.strip()
                cat_votes[c] = cat_votes.get(c, 0) + w
        
        # Sort
        top_names = sorted(name_votes.items(), key=lambda x: x[1], reverse=True)[:5]
        top_cats = sorted(cat_votes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return top_names, top_cats

# gradio interface
try:
    engine = SearchEngine()
except Exception as e:
    print(f"Error initializing engine: {e}")
    engine = None

def app_logic(text_input, image_input):
    if engine is None:
        return "System not initialized. Check files.", []

    if image_input is not None:
        # Image mode
        top_names, top_cats = engine.classify(image_input)
        similar_imgs = engine.search(query_image=image_input, top_k=5)
        
        if not similar_imgs:
             return "No similar images found or images directory is empty.", []

        # Format text
        names_str = "\n".join([f"{n[0]} ({n[1]:.2f})" for n in top_names])
        cats_str = "\n".join([f"{c[0]} ({c[1]:.2f})" for c in top_cats])
        info = f"**Top 5 Names:**\n{names_str}\n\n**Top 5 Categories:**\n{cats_str}"
        
        gallery = [(r['image'], r['caption']) for r in similar_imgs]
        return info, gallery
        
    elif text_input:
        # Text mode
        results = engine.search(query_text=text_input, top_k=5)
        if not results:
            return "No results found. Ensure 'images' folder is populated.", []
            
        gallery = [(r['image'], r['caption']) for r in results]
        return f"Search results for: '{text_input}'", gallery
    
    return "Please upload an image or enter text.", []

# Interface Design
with gr.Blocks(title="AI Landmark Search") as demo:
    gr.Markdown("# AI Landmark Search")
    gr.Markdown("Search for landmarks by text or image.")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp_text = gr.Textbox(label="Search by text", placeholder="e.g., Red brick church")
            inp_img = gr.Image(label="Search/Classify by photo", type="pil")
            btn = gr.Button("Search", variant="primary")
        
        with gr.Column(scale=2):
            out_text = gr.Markdown(label="Analysis Results")
            out_gallery = gr.Gallery(label="Found Objects", columns=3, height=600)

    btn.click(app_logic, inputs=[inp_text, inp_img], outputs=[out_text, out_gallery])

if __name__ == "__main__":
    # Allow overriding port via environment variable, default 7860
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
