import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def _compute_matrix(vectors: np.ndarray, algo: str) -> np.ndarray:
    algo_upper = algo.upper().replace(" ", "_")
    if algo_upper == "COSINE":
        return cosine_similarity(vectors)
    elif algo_upper == "EUCLIDEAN":
        dist = euclidean_distances(vectors)
        return 1.0 / (1.0 + dist)
    elif algo_upper in ("INNER_PRODUCT", "INNER PRODUCT"):
        return vectors @ vectors.T
    else:
        return cosine_similarity(vectors)


def compute_pairwise(
    vectors: list[list[float]], names: list[str], algo: str
) -> list[dict]:
    arr = np.array(vectors, dtype=np.float32)
    matrix = _compute_matrix(arr, algo)
    results = []
    for i in range(len(names)):
        for j in range(len(names)):
            results.append({
                "classifier_a": names[i],
                "classifier_b": names[j],
                "cosine_similarity": round(float(matrix[i, j]), 6),
            })
    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return results


def compute_cross(
    image_vectors: list[list[float]],
    classifier_vectors: list[list[float]],
    image_names: list[str],
    classifier_names: list[str],
    algo: str,
) -> np.ndarray:
    img_arr = np.array(image_vectors, dtype=np.float32)
    cls_arr = np.array(classifier_vectors, dtype=np.float32)

    algo_upper = algo.upper().replace(" ", "_")
    if algo_upper == "COSINE":
        return cosine_similarity(img_arr, cls_arr)
    elif algo_upper == "EUCLIDEAN":
        dist = euclidean_distances(img_arr, cls_arr)
        return 1.0 / (1.0 + dist)
    elif algo_upper in ("INNER_PRODUCT", "INNER PRODUCT"):
        return img_arr @ cls_arr.T
    else:
        return cosine_similarity(img_arr, cls_arr)
