"""
Intelligent Model Router

AI-based routing to select the optimal model for each document.
Unlike rule-based routing (by file extension), this analyzes
document characteristics to make intelligent decisions.

Factors considered:
1. Layout complexity (tables, forms, free text)
2. Text density (sparse vs dense)
3. Handwriting presence
4. Image quality (noise, skew, resolution)
5. Document type classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        ViTForImageClassification,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ModelType(Enum):
    """Available model types for document processing."""
    LAYOUTLMV3 = "layoutlmv3"  # Complex structured documents
    DONUT = "donut"            # OCR-free, noisy documents
    TROCR = "trocr"            # Handwriting recognition
    LLM_VISION = "llm_vision"  # Complex reasoning (GPT-4V/Claude)
    HYBRID = "hybrid"          # Combine multiple models


@dataclass
class DocumentCharacteristics:
    """Analyzed characteristics of a document."""
    layout_complexity: float      # 0-1, higher = more complex layout
    text_density: float           # 0-1, higher = more text
    handwriting_probability: float  # 0-1, probability of handwriting
    noise_level: float            # 0-1, higher = more noise
    table_probability: float      # 0-1, probability of tables
    form_probability: float       # 0-1, probability of form structure
    document_type: str            # Classified document type
    confidence: float             # Confidence in classification


@dataclass
class RoutingDecision:
    """Model routing decision."""
    primary_model: ModelType
    fallback_model: Optional[ModelType]
    confidence: float
    reasoning: str
    characteristics: DocumentCharacteristics


class DocumentAnalyzer(nn.Module):
    """
    CNN-based document characteristic analyzer.

    Extracts features relevant for model selection:
    - Layout structure
    - Text density
    - Handwriting detection
    - Noise estimation
    """

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()

        # Use pre-trained backbone
        if backbone == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            hidden_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Multi-task heads
        self.layout_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.handwriting_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.table_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze document characteristics.

        Args:
            x: Image tensor (B, C, H, W)

        Returns:
            Dict of characteristic scores
        """
        # Extract features
        features = self.backbone(x)
        features = features.flatten(1)

        return {
            'layout_complexity': self.layout_head(features),
            'text_density': self.density_head(features),
            'handwriting_probability': self.handwriting_head(features),
            'noise_level': self.noise_head(features),
            'table_probability': self.table_head(features),
        }


class DocumentClassifier:
    """
    Document type classifier using Vision Transformer.

    Classifies documents into categories:
    - Invoice
    - Purchase Order
    - Receipt
    - Form
    - Letter
    - Report
    - Other
    """

    DOCUMENT_TYPES = [
        "invoice", "purchase_order", "receipt", "form",
        "letter", "report", "contract", "other"
    ]

    def __init__(
        self,
        model_name: str = "microsoft/dit-base-finetuned-rvlcdip",
        device: str = None,
    ):
        """
        Initialize document classifier.

        Args:
            model_name: HuggingFace model for document classification
                Default uses DiT fine-tuned on RVL-CDIP dataset
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if HAS_TRANSFORMERS:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.processor = None
            self.model = None

    def classify(self, image: Image.Image) -> Tuple[str, float]:
        """
        Classify document type.

        Returns:
            Tuple of (document_type, confidence)
        """
        if self.model is None:
            return "other", 0.5

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()

        # Map to document type
        if hasattr(self.model.config, 'id2label'):
            doc_type = self.model.config.id2label[pred_idx].lower()
        else:
            doc_type = self.DOCUMENT_TYPES[pred_idx % len(self.DOCUMENT_TYPES)]

        return doc_type, confidence


class ModelRouter:
    """
    Intelligent model router for document processing.

    Makes AI-based decisions about which model(s) to use
    based on document characteristics.
    """

    def __init__(
        self,
        device: str = None,
        use_analyzer: bool = True,
        use_classifier: bool = True,
    ):
        """
        Initialize model router.

        Args:
            device: Device to run on
            use_analyzer: Use CNN analyzer for characteristics
            use_classifier: Use ViT classifier for document type
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.analyzer = None
        self.classifier = None

        if use_analyzer:
            self.analyzer = DocumentAnalyzer()
            self.analyzer.to(self.device)
            self.analyzer.eval()

        if use_classifier:
            try:
                self.classifier = DocumentClassifier(device=self.device)
            except Exception as e:
                print(f"Warning: Could not load classifier: {e}")
                self.classifier = None

        # Routing rules (learned or configured)
        self.routing_config = {
            'handwriting_threshold': 0.6,
            'layout_complexity_threshold': 0.7,
            'noise_threshold': 0.5,
            'table_threshold': 0.6,
        }

    def preprocess_image(
        self,
        image: Union[Image.Image, Path, str],
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Preprocess image for analysis."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)

        return image, tensor

    def analyze_document(
        self,
        image: Union[Image.Image, Path, str],
    ) -> DocumentCharacteristics:
        """
        Analyze document characteristics.

        This is the core of intelligent routing - understanding
        what kind of document we're dealing with.
        """
        image, tensor = self.preprocess_image(image)

        # Get characteristic scores
        if self.analyzer is not None:
            with torch.no_grad():
                scores = self.analyzer(tensor)

            layout_complexity = scores['layout_complexity'].item()
            text_density = scores['text_density'].item()
            handwriting_prob = scores['handwriting_probability'].item()
            noise_level = scores['noise_level'].item()
            table_prob = scores['table_probability'].item()
        else:
            # Use simple heuristics if analyzer not available
            scores = self._heuristic_analysis(image)
            layout_complexity = scores['layout_complexity']
            text_density = scores['text_density']
            handwriting_prob = scores['handwriting_probability']
            noise_level = scores['noise_level']
            table_prob = scores['table_probability']

        # Classify document type
        if self.classifier is not None:
            doc_type, type_confidence = self.classifier.classify(image)
        else:
            doc_type, type_confidence = "other", 0.5

        # Estimate form probability
        form_prob = self._estimate_form_probability(image)

        return DocumentCharacteristics(
            layout_complexity=layout_complexity,
            text_density=text_density,
            handwriting_probability=handwriting_prob,
            noise_level=noise_level,
            table_probability=table_prob,
            form_probability=form_prob,
            document_type=doc_type,
            confidence=type_confidence,
        )

    def _heuristic_analysis(self, image: Image.Image) -> Dict[str, float]:
        """
        Simple heuristic analysis when model not available.

        Uses image processing techniques to estimate characteristics.
        """
        import numpy as np

        img_array = np.array(image.convert('L'))  # Grayscale

        # Text density: ratio of dark pixels
        threshold = 128
        dark_ratio = (img_array < threshold).mean()
        text_density = min(dark_ratio * 5, 1.0)  # Scale

        # Noise estimation: high frequency components
        from scipy import ndimage
        laplacian = ndimage.laplace(img_array.astype(float))
        noise_level = min(np.abs(laplacian).mean() / 50, 1.0)

        # Layout complexity: edge density variation
        edges = ndimage.sobel(img_array.astype(float))
        edge_std = edges.std()
        layout_complexity = min(edge_std / 50, 1.0)

        # Table probability: horizontal/vertical line detection
        # (simplified - would use Hough transform in production)
        table_prob = 0.3  # Default moderate

        return {
            'layout_complexity': layout_complexity,
            'text_density': text_density,
            'handwriting_probability': 0.0,  # Can't detect without model
            'noise_level': noise_level,
            'table_probability': table_prob,
        }

    def _estimate_form_probability(self, image: Image.Image) -> float:
        """
        Estimate probability that document is a form.

        Forms typically have:
        - Regular grid structure
        - Labels followed by blank spaces
        - Checkbox patterns
        """
        # Simplified heuristic
        # Production would use a trained detector
        return 0.5

    def route(
        self,
        image: Union[Image.Image, Path, str],
    ) -> RoutingDecision:
        """
        Make routing decision for document.

        This is the main intelligence - deciding which model(s)
        will best handle this specific document.
        """
        # Analyze document
        characteristics = self.analyze_document(image)

        # Decision logic
        primary_model = ModelType.LAYOUTLMV3
        fallback_model = None
        reasoning_parts = []

        # Rule 1: Handwriting detection
        if characteristics.handwriting_probability > self.routing_config['handwriting_threshold']:
            primary_model = ModelType.TROCR
            fallback_model = ModelType.LLM_VISION
            reasoning_parts.append(
                f"Handwriting detected ({characteristics.handwriting_probability:.2f})"
            )

        # Rule 2: High noise - use OCR-free model
        elif characteristics.noise_level > self.routing_config['noise_threshold']:
            primary_model = ModelType.DONUT
            fallback_model = ModelType.LLM_VISION
            reasoning_parts.append(
                f"High noise level ({characteristics.noise_level:.2f}), using OCR-free model"
            )

        # Rule 3: Complex layout with tables
        elif (characteristics.layout_complexity > self.routing_config['layout_complexity_threshold'] and
              characteristics.table_probability > self.routing_config['table_threshold']):
            primary_model = ModelType.LAYOUTLMV3
            fallback_model = ModelType.HYBRID
            reasoning_parts.append(
                f"Complex layout ({characteristics.layout_complexity:.2f}) "
                f"with tables ({characteristics.table_probability:.2f})"
            )

        # Rule 4: Simple document
        elif characteristics.layout_complexity < 0.3 and characteristics.text_density > 0.7:
            primary_model = ModelType.DONUT
            reasoning_parts.append("Simple document with high text density")

        # Rule 5: Form structure
        elif characteristics.form_probability > 0.7:
            primary_model = ModelType.LAYOUTLMV3
            reasoning_parts.append(f"Form detected ({characteristics.form_probability:.2f})")

        # Default: Use hybrid for uncertain cases
        else:
            primary_model = ModelType.HYBRID
            fallback_model = ModelType.LLM_VISION
            reasoning_parts.append("Using hybrid approach for best coverage")

        # Calculate confidence
        confidence = characteristics.confidence

        return RoutingDecision(
            primary_model=primary_model,
            fallback_model=fallback_model,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Default routing",
            characteristics=characteristics,
        )

    def get_model_config(self, decision: RoutingDecision) -> Dict:
        """
        Get configuration for selected model.

        Returns model-specific parameters based on document characteristics.
        """
        config = {
            'model_type': decision.primary_model.value,
            'fallback': decision.fallback_model.value if decision.fallback_model else None,
        }

        # Model-specific configuration
        if decision.primary_model == ModelType.LAYOUTLMV3:
            config.update({
                'apply_ocr': True,
                'max_length': 512,
                'use_visual_features': True,
            })

        elif decision.primary_model == ModelType.DONUT:
            config.update({
                'max_length': 512,
                'num_beams': 4,
            })

        elif decision.primary_model == ModelType.TROCR:
            config.update({
                'enhance_image': True,
                'num_beams': 4,
                'detect_lines': decision.characteristics.text_density < 0.5,
            })

        elif decision.primary_model == ModelType.LLM_VISION:
            config.update({
                'model': 'gpt-4o',  # or claude-3-sonnet
                'temperature': 0.1,
                'max_tokens': 2000,
            })

        elif decision.primary_model == ModelType.HYBRID:
            config.update({
                'models': ['layoutlmv3', 'donut'],
                'fusion': 'weighted_average',
            })

        return config
