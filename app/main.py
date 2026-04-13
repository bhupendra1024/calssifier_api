import base64
import io
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from PIL import Image

from app import db, embedder, similarity
from app.auth import get_current_user
from app.schemas import (
    ClassificationAnalyzerRequest,
    CreateClassifierConfigRequest,
    ImageComparisonRequest,
    UpdateClassifierConfigRequest,
)

app = FastAPI(title="Classification Config API")


@app.on_event("startup")
def startup():
    embedder.preload_model()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_config_direct(org_name: str, classifier_config_model: str):
    cls_cfg = db.get_classifier_config(org_name, classifier_config_model)
    if not cls_cfg:
        raise HTTPException(status_code=404, detail="Classifier config not found for this org and model")

    llm_type = cls_cfg.get("llm_type")
    if not llm_type:
        raise HTTPException(status_code=400, detail="llm_type not configured in classifier_config_json")

    matching_algo = cls_cfg.get("matching_algo", "COSINE")
    return llm_type, matching_algo


# ── API 1: Create ───────────────────────────────────────────────────────────

@app.post("/create_classification_config_details")
def create_classification_config_details(
    req: CreateClassifierConfigRequest,
    user_id: str = Depends(get_current_user),
):
    llm_type, _ = _get_config_direct(req.org_name, req.classifier_config_model)

    # Validate none of the classifiers already exist
    existing = [
        item.classifier for item in req.classifiers
        if db.classifier_detail_exists(req.org_name, req.classifier_config_model, item.classifier)
    ]
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Classifiers already exist for model '{req.classifier_config_model}': {existing}",
        )

    vectors = embedder.generate_text_embeddings(
        [item.text_data for item in req.classifiers], llm_type
    )

    count = 0
    for item, vec in zip(req.classifiers, vectors):
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
        db.insert_classifier_detail(
            org_name=req.org_name,
            classifier_config_model=req.classifier_config_model,
            classifier=item.classifier,
            text_data=item.text_data,
            vector_data=vec_str,
            created_by=user_id,
        )
        count += 1

    return {"status": "SUCCESS", "message": f"{count} classifier(s) created"}


# ── API 2: Update ───────────────────────────────────────────────────────────

@app.put("/update_classifier_config_details")
def update_classifier_config_details(
    req: UpdateClassifierConfigRequest,
    user_id: str = Depends(get_current_user),
):
    llm_type, _ = _get_config_direct(req.org_name, req.classifier_config_model)

    # Validate all classifiers exist first
    for item in req.classifiers:
        if not db.classifier_detail_exists(req.org_name, req.classifier_config_model, item.classifier):
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{item.classifier}' not found for model '{req.classifier_config_model}'",
            )

    vectors = embedder.generate_text_embeddings(
        [item.text_data for item in req.classifiers], llm_type
    )

    count = 0
    for item, vec in zip(req.classifiers, vectors):
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
        db.update_classifier_detail(
            org_name=req.org_name,
            classifier_config_model=req.classifier_config_model,
            classifier=item.classifier,
            text_data=item.text_data,
            vector_data=vec_str,
            modified_by=user_id,
        )
        count += 1

    return {"status": "SUCCESS", "message": f"{count} classifier(s) updated"}


# ── API 3: Classification Analyzer ──────────────────────────────────────────

@app.post("/classification_analyzer")
def classification_analyzer(
    req: ClassificationAnalyzerRequest,
    user_id: str = Depends(get_current_user),
):
    _, matching_algo = _get_config_direct(req.org_name, req.classifier_config_model)

    # Fetch classifier_config_details (vectors)
    details = db.get_classifier_details(req.org_name, req.classifier_config_model)
    if not details:
        raise HTTPException(status_code=404, detail="No classifier vectors found for this model")

    names = [d["classifier"] for d in details]
    vectors = [d["vector_data"] for d in details]

    # vector_data comes from pgvector as a string or list — normalize
    parsed_vectors = []
    for v in vectors:
        if isinstance(v, str):
            v = v.strip("[]")
            parsed_vectors.append([float(x) for x in v.split(",")])
        elif isinstance(v, list):
            parsed_vectors.append([float(x) for x in v])
        else:
            parsed_vectors.append(list(v))

    results = similarity.compute_pairwise(parsed_vectors, names, matching_algo)

    return {
        "classifier_config_model": req.classifier_config_model,
        "results": results,
    }


# ── API 4: Image Comparison ─────────────────────────────────────────────────

@app.post("/image_comparison")
def image_comparison(
    req: ImageComparisonRequest,
    user_id: str = Depends(get_current_user),
):
    try:
        print(f"Received {len(req.images)} images for comparison")

        llm_type, matching_algo = _get_config_direct(req.org_name, req.classifier_config_model)

        # Decode base64 images
        pil_images = []
        image_names = []

        for img in req.images:
            try:
                data = base64.b64decode(img.image_base64)
            except Exception as e:
                print(f"Base64 decode failed for '{img.image_name}': {e}")
                raise HTTPException(status_code=400, detail=f"Invalid base64 for image '{img.image_name}'")
            pil_images.append(Image.open(io.BytesIO(data)).convert("RGB"))
            print(f"Decoded image '{img.image_name}' successfully")
            image_names.append(img.image_name)
            del data

        # Fetch classifier vectors
        details = db.get_classifier_details(req.org_name, req.classifier_config_model)
        if not details:
            raise HTTPException(status_code=404, detail="No classifier vectors found for this model")

        cls_names = [d["classifier"] for d in details]
        cls_vectors = []
        for v in [d["vector_data"] for d in details]:
            if isinstance(v, str):
                v = v.strip("[]")
                cls_vectors.append([float(x) for x in v.split(",")])
            elif isinstance(v, list):
                cls_vectors.append([float(x) for x in v])
            else:
                cls_vectors.append(list(v))

        # Encode images
        print(f"Encoding {len(pil_images)} images with model '{llm_type}'")
        img_vectors = embedder.generate_image_embeddings(pil_images, llm_type)

        # Compute similarity matrix: (num_images x num_classifiers)
        score_matrix = similarity.compute_cross(
            img_vectors, cls_vectors, image_names, cls_names, matching_algo
        )

        # IMG-TO-CLS: for each image, score against all classifiers
        results = []
        for i, img_name in enumerate(image_names):
            scores = []
            for j, cls_name in enumerate(cls_names):
                s = round(float(score_matrix[i, j]), 6)
                scores.append({"classifier": cls_name, "score": s})

            scores.sort(key=lambda x: x["score"], reverse=True)
            results.append({"image": img_name, "scores": scores})

        print(f"Image comparison completed: {len(results)} results")
        return {
            "classifier_config_model": req.classifier_config_model,
            "results": results,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
