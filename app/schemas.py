from pydantic import BaseModel


# ── Shared ──────────────────────────────────────────────────────────────────

class ClassifierItem(BaseModel):
    c_name: str
    c_text: str


# ── API 1: Create ───────────────────────────────────────────────────────────

class CreateClassifierConfigRequest(BaseModel):
    key: str
    classifier_config_model: str
    classifier_list: list[ClassifierItem]


# ── API 2: Update ───────────────────────────────────────────────────────────

class UpdateClassifierConfigRequest(BaseModel):
    key: str
    classifier_config_model: str
    updated_classifiers: list[ClassifierItem]


# ── API 4: Image Comparison ─────────────────────────────────────────────────

class ImageItem(BaseModel):
    image_name: str
    image_base64: str


class ImageComparisonRequest(BaseModel):
    key: str
    classifier_config_model: str
    images: list[ImageItem]


# ── API 3: Analyzer ─────────────────────────────────────────────────────────

class ClassificationAnalyzerRequest(BaseModel):
    key: str
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
