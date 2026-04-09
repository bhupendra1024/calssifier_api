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

def _get_configs(key: str, api_name: str, classifier_config_model: str):
    api_cfg = db.get_api_config(key, api_name) or {}

    cls_cfg = db.get_classifier_config(key, classifier_config_model)
    if not cls_cfg:
        raise HTTPException(status_code=404, detail="Classifier config model not found")

    llm_type = cls_cfg.get("llm_type")
    if not llm_type:
        raise HTTPException(status_code=400, detail="llm_type not configured in classifier_config_json")

    matching_algo = cls_cfg.get("matching_algo", "COSINE")
    return api_cfg, llm_type, matching_algo


def _apply_output_config(results: list[dict], output_config: dict) -> list[dict]:
    if not output_config:
        return results

    # selection_threshold: filter by score
    threshold = output_config.get("selection_threshold", "ALL")
    if threshold != "ALL":
        try:
            t = float(threshold)
            results = [r for r in results if r.get("cosine_similarity", r.get("score", 0)) >= t]
        except (ValueError, TypeError):
            pass

    # display_threshold: limit count
    display = output_config.get("display_threshold", "ALL")
    if display != "ALL":
        try:
            n = int(display)
            results = results[:n]
        except (ValueError, TypeError):
            pass

    # confidence_score_flag: remove scores if false
    if not output_config.get("confidence_score_flag", True):
        for r in results:
            r.pop("cosine_similarity", None)
            r.pop("score", None)

    return results


# ── API 1: Create ───────────────────────────────────────────────────────────

@app.post("/create_classification_config_details")
def create_classification_config_details(
    req: CreateClassifierConfigRequest,
    user_id: str = Depends(get_current_user),
):
    api_cfg, llm_type, _ = _get_configs(
        req.key, "create_classification_config_details", req.classifier_config_model
    )

    # Validate none of the classifiers already exist
    existing = [
        item.c_name for item in req.classifier_list
        if db.classifier_detail_exists(req.key, req.classifier_config_model, item.c_name)
    ]
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Classifiers already exist for model '{req.classifier_config_model}': {existing}",
        )

    texts = [item.c_text for item in req.classifier_list]
    names = [item.c_name for item in req.classifier_list]
    vectors = embedder.generate_text_embeddings(texts, llm_type)

    count = 0
    for name, text, vec in zip(names, texts, vectors):
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
        db.insert_classifier_detail(
            key=req.key,
            classifier_config_model=req.classifier_config_model,
            classifier=name,
            text_data=text,
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
    api_cfg, llm_type, _ = _get_configs(
        req.key, "update_classifier_config_details", req.classifier_config_model
    )

    # Validate all classifiers exist first
    for item in req.updated_classifiers:
        if not db.classifier_detail_exists(req.key, req.classifier_config_model, item.c_name):
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{item.c_name}' not found for model '{req.classifier_config_model}'",
            )

    texts = [item.c_text for item in req.updated_classifiers]
    names = [item.c_name for item in req.updated_classifiers]
    vectors = embedder.generate_text_embeddings(texts, llm_type)

    count = 0
    for name, text, vec in zip(names, texts, vectors):
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
        db.update_classifier_detail(
            key=req.key,
            classifier_config_model=req.classifier_config_model,
            classifier=name,
            text_data=text,
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
    api_cfg, llm_type, matching_algo = _get_configs(
        req.key, "classification_analyzer", req.classifier_config_model
    )

    details = db.get_classifier_details(req.key, req.classifier_config_model)
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

    output_config = api_cfg.get("output_config", {})
    results = _apply_output_config(results, output_config)

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

        api_cfg, llm_type, matching_algo = _get_configs(
            req.key, "image_comparison", req.classifier_config_model
        )

        output_config = api_cfg.get("output_config", {})

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
        details = db.get_classifier_details(req.key, req.classifier_config_model)
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

        # Format response based on output_mode
        output_mode = output_config.get("output_mode", "IMG-TO-CLS")
        confidence_flag = output_config.get("confidence_score_flag", True)
        selection_threshold = output_config.get("selection_threshold", "ALL")
        display_threshold = output_config.get("display_threshold", "ALL")

        results = []

        if output_mode == "CLS-TO-IMG":
            for j, cls_name in enumerate(cls_names):
                scores = []
                for i, img_name in enumerate(image_names):
                    s = round(float(score_matrix[i, j]), 6)
                    entry = {"image": img_name}
                    if confidence_flag:
                        entry["score"] = s
                    scores.append((s, entry))

                scores.sort(key=lambda x: x[0], reverse=True)

                if selection_threshold != "ALL":
                    try:
                        t = float(selection_threshold)
                        scores = [(sc, e) for sc, e in scores if sc >= t]
                    except (ValueError, TypeError):
                        pass

                if display_threshold != "ALL":
                    try:
                        n = int(display_threshold)
                        scores = scores[:n]
                    except (ValueError, TypeError):
                        pass

                results.append({
                    "classifier": cls_name,
                    "scores": [entry for _, entry in scores],
                })
        else:
            for i, img_name in enumerate(image_names):
                scores = []
                for j, cls_name in enumerate(cls_names):
                    s = round(float(score_matrix[i, j]), 6)
                    entry = {"classifier": cls_name}
                    if confidence_flag:
                        entry["score"] = s
                    scores.append((s, entry))

                scores.sort(key=lambda x: x[0], reverse=True)

                if selection_threshold != "ALL":
                    try:
                        t = float(selection_threshold)
                        scores = [(sc, e) for sc, e in scores if sc >= t]
                    except (ValueError, TypeError):
                        pass

                if display_threshold != "ALL":
                    try:
                        n = int(display_threshold)
                        scores = scores[:n]
                    except (ValueError, TypeError):
                        pass

                results.append({
                    "image": img_name,
                    "scores": [entry for _, entry in scores],
                })

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
