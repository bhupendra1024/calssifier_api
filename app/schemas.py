from pydantic import BaseModel


# ── Shared ──────────────────────────────────────────────────────────────────

class ClassifierItem(BaseModel):
    classifier: str
    text_data: str


# ── API 1: Create ───────────────────────────────────────────────────────────

class CreateClassifierConfigRequest(BaseModel):
    org_name: str
    classifier_config_model: str
    classifiers: list[ClassifierItem]


# ── API 2: Update ───────────────────────────────────────────────────────────

class UpdateClassifierConfigRequest(BaseModel):
    org_name: str
    classifier_config_model: str
    classifiers: list[ClassifierItem]


# ── API 4: Image Comparison ─────────────────────────────────────────────────

class ImageItem(BaseModel):
    image_name: str
    image_base64: str


class ImageComparisonRequest(BaseModel):
    org_name: str
    classifier_config_model: str
    images: list[ImageItem]


# ── API 3: Analyzer ─────────────────────────────────────────────────────────

class ClassificationAnalyzerRequest(BaseModel):
    org_name: str
    classifier_config_model: str


# ── Responses ───────────────────────────────────────────────────────────────

class PairScore(BaseModel):
    classifier_a: str
    classifier_b: str
    cosine_similarity: float


class AnalyzerResponse(BaseModel):
    classifier_config_model: str
    results: list[PairScore]


class ImageScore(BaseModel):
    classifier: str
    score: float


class ImageResult(BaseModel):
    image: str
    scores: list[ImageScore]


class ClassifierResult(BaseModel):
    classifier: str
    scores: list[dict]


class ImageComparisonResponse(BaseModel):
    classifier_config_model: str
    results: list[dict]
