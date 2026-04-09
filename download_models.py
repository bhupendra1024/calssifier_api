"""Pre-download CLIP model weights during Docker build."""
from sentence_transformers import SentenceTransformer

MODELS = [
    "clip-ViT-B-32",
]

if __name__ == "__main__":
    for model_name in MODELS:
        print(f"Downloading {model_name}...")
        SentenceTransformer(model_name)
        print(f"  {model_name} ready.")
    print("All models downloaded.")
