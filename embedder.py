# import hashlib
# import numpy as np
# import db

# MODEL_NAME = "clip-ViT-B-32"


# def _hash(text: str) -> str:
#     return hashlib.sha256(text.encode()).hexdigest()


# def get_embeddings(contexts: dict, model) -> dict:
#     """
#     Returns {context_name: np.ndarray} for all contexts.
#     Fetches from DB if hash matches; otherwise encodes and stores.
#     """
#     cached = db.fetch_cached_embeddings(MODEL_NAME)

#     result = {}
#     to_encode = {}

#     for name, description in contexts.items():
#         ctx_hash = _hash(description)
#         if name in cached and cached[name][0] == ctx_hash:
#             result[name] = np.array(cached[name][1], dtype=np.float32)
#         else:
#             to_encode[name] = (description, ctx_hash)

#     if to_encode:
#         names = list(to_encode.keys())
#         texts = [to_encode[n][0] for n in names]
#         vectors = model.encode(texts, convert_to_numpy=True)

#         new_rows = []
#         for name, vector in zip(names, vectors):
#             ctx_hash = to_encode[name][1]
#             result[name] = vector
#             new_rows.append((name, ctx_hash, vector.tolist()))

#         db.save_embeddings(new_rows, MODEL_NAME)
#         print(f"Encoded and stored {len(new_rows)} new/updated context(s).")
#     else:
#         print("All embeddings loaded from cache.")

#     return result
