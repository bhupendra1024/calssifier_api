from sentence_transformers import SentenceTransformer
from PIL import Image

DEFAULT_MODEL = "clip-ViT-B-32"
_model_cache: dict[str, SentenceTransformer] = {}


def preload_model(model_name: str = DEFAULT_MODEL):
    print(f"Pre-loading model '{model_name}'...")
    get_model(model_name)
    print(f"Model '{model_name}' loaded and ready.")


def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def generate_text_embeddings(
    texts: list[str], model_name: str
) -> list[list[float]]:
    model = get_model(model_name)
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def generate_image_embeddings(
    images: list[Image.Image], model_name: str, batch_size: int = 5
) -> list[list[float]]:
    model = get_model(model_name)
    all_vectors = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        vectors = model.encode(batch, convert_to_numpy=True)
        all_vectors.extend(v.tolist() for v in vectors)
    return all_vectors
