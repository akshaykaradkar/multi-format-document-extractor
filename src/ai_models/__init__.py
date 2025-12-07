"""
AI Models Module

This module contains state-of-the-art AI models for document understanding:
- LayoutLMv3: Tri-modal transformer for structured documents
- Donut: OCR-free document understanding
- TrOCR: Transformer-based handwriting recognition
- ModelRouter: AI-based model selection
"""

from .document_encoder import DocumentEncoder, LayoutLMv3Encoder
from .ocr_free_model import DonutExtractor
from .handwriting_model import TrOCRExtractor
from .model_router import ModelRouter
from .ensemble import ModelEnsemble

__all__ = [
    "DocumentEncoder",
    "LayoutLMv3Encoder",
    "DonutExtractor",
    "TrOCRExtractor",
    "ModelRouter",
    "ModelEnsemble",
]
