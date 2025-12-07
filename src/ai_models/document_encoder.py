"""
Document Encoder using LayoutLMv3

LayoutLMv3 is a tri-modal transformer that jointly models:
1. Text embeddings (from OCR)
2. Layout embeddings (2D position)
3. Visual embeddings (image patches)

This enables understanding document structure, not just text content.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Conditional imports for flexibility
try:
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3ForSequenceClassification,
        AutoProcessor,
        AutoModelForTokenClassification,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


@dataclass
class BoundingBox:
    """Normalized bounding box (0-1000 scale for LayoutLM)."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_normalized(cls, box: List[float], width: int, height: int) -> "BoundingBox":
        """Convert from normalized (0-1) to LayoutLM scale (0-1000)."""
        return cls(
            x_min=int(box[0] * 1000 / width * width),
            y_min=int(box[1] * 1000 / height * height),
            x_max=int(box[2] * 1000 / width * width),
            y_max=int(box[3] * 1000 / height * height),
        )


@dataclass
class OCRResult:
    """Result from OCR processing."""
    words: List[str]
    boxes: List[BoundingBox]
    confidence: List[float]


@dataclass
class ExtractionResult:
    """Result from document field extraction."""
    fields: Dict[str, str]
    confidence_scores: Dict[str, float]
    token_predictions: Optional[List[str]] = None
    raw_logits: Optional[torch.Tensor] = None


class DocumentEncoder(ABC):
    """Abstract base class for document encoders."""

    @abstractmethod
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Encode document image to embeddings."""
        pass

    @abstractmethod
    def extract_fields(self, image: Image.Image) -> ExtractionResult:
        """Extract structured fields from document."""
        pass


class LayoutLMv3Encoder(DocumentEncoder):
    """
    LayoutLMv3-based document encoder for structured field extraction.

    This is the state-of-the-art (2024) approach for document understanding,
    achieving 83.37 ANLS on DocVQA benchmark.

    Key innovations:
    1. Tri-modal fusion (text + layout + vision)
    2. Unified pre-training on text-image alignment
    3. Spatial-aware self-attention
    """

    # Field labels for token classification
    FIELD_LABELS = [
        "O",  # Outside any field
        "B-ORDER_ID", "I-ORDER_ID",
        "B-CLIENT_NAME", "I-CLIENT_NAME",
        "B-ORDER_DATE", "I-ORDER_DATE",
        "B-DELIVERY_DATE", "I-DELIVERY_DATE",
        "B-PRODUCT_CODE", "I-PRODUCT_CODE",
        "B-DESCRIPTION", "I-DESCRIPTION",
        "B-QUANTITY", "I-QUANTITY",
        "B-UNIT_PRICE", "I-UNIT_PRICE",
        "B-TOTAL_PRICE", "I-TOTAL_PRICE",
        "B-ORDER_TOTAL", "I-ORDER_TOTAL",
        "B-SPECIAL_INSTRUCTIONS", "I-SPECIAL_INSTRUCTIONS",
    ]

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        device: str = None,
        use_ocr: bool = True,
    ):
        """
        Initialize LayoutLMv3 encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu/mps)
            use_ocr: Whether to run OCR (False if pre-extracted)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers"
            )

        self.device = device or self._get_device()
        self.use_ocr = use_ocr
        self.model_name = model_name

        # Initialize processor and model
        self._init_model()

    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_model(self):
        """Initialize the LayoutLMv3 model and processor."""
        print(f"Loading LayoutLMv3 from {self.model_name}...")

        # Processor handles image preprocessing and tokenization
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.model_name,
            apply_ocr=self.use_ocr,  # Let processor handle OCR
        )

        # Model for token classification (field extraction)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.FIELD_LABELS),
            id2label={i: l for i, l in enumerate(self.FIELD_LABELS)},
            label2id={l: i for i, l in enumerate(self.FIELD_LABELS)},
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on {self.device}")

    def preprocess_image(self, image: Union[Image.Image, Path, str]) -> Image.Image:
        """Load and preprocess document image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def run_ocr(self, image: Image.Image) -> OCRResult:
        """
        Run OCR to extract text and bounding boxes.

        Uses Tesseract by default, but can be swapped for other engines.
        """
        if not HAS_TESSERACT:
            raise ImportError(
                "pytesseract required for OCR. Install with: "
                "pip install pytesseract"
            )

        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )

        words = []
        boxes = []
        confidences = []

        width, height = image.size

        for i, word in enumerate(ocr_data["text"]):
            if word.strip():  # Skip empty strings
                words.append(word)

                # Convert to LayoutLM normalized coordinates (0-1000)
                x = ocr_data["left"][i]
                y = ocr_data["top"][i]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]

                box = BoundingBox(
                    x_min=int(x * 1000 / width),
                    y_min=int(y * 1000 / height),
                    x_max=int((x + w) * 1000 / width),
                    y_max=int((y + h) * 1000 / height),
                )
                boxes.append(box)

                conf = ocr_data["conf"][i]
                confidences.append(conf / 100.0 if conf > 0 else 0.0)

        return OCRResult(words=words, boxes=boxes, confidence=confidences)

    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Encode document image to contextualized embeddings.

        Returns:
            Tensor of shape (seq_len, hidden_size) containing
            tri-modal fused representations.
        """
        image = self.preprocess_image(image)

        # Process image (includes OCR if enabled)
        encoding = self.processor(
            image,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Get embeddings from model
        with torch.no_grad():
            outputs = self.model.layoutlmv3(**encoding, output_hidden_states=True)

        # Return last hidden state
        return outputs.hidden_states[-1]

    def extract_fields(
        self,
        image: Union[Image.Image, Path, str],
        ocr_result: Optional[OCRResult] = None,
    ) -> ExtractionResult:
        """
        Extract structured fields from document image.

        This uses token classification to identify field boundaries
        and values in the document.

        Args:
            image: Document image
            ocr_result: Pre-computed OCR result (optional)

        Returns:
            ExtractionResult with extracted fields and confidence scores
        """
        image = self.preprocess_image(image)

        # Process image
        if ocr_result and not self.use_ocr:
            # Use provided OCR result
            encoding = self.processor(
                image,
                text=ocr_result.words,
                boxes=[b.to_list() for b in ocr_result.boxes],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        else:
            # Let processor handle OCR
            encoding = self.processor(
                image,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Decode predictions to fields
        fields, confidence_scores = self._decode_predictions(
            predictions[0],
            logits[0],
            encoding,
        )

        return ExtractionResult(
            fields=fields,
            confidence_scores=confidence_scores,
            token_predictions=[
                self.FIELD_LABELS[p.item()] for p in predictions[0]
            ],
            raw_logits=logits,
        )

    def _decode_predictions(
        self,
        predictions: torch.Tensor,
        logits: torch.Tensor,
        encoding: Dict,
    ) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Decode token-level predictions into field values.

        Uses BIO tagging scheme:
        - B-FIELD: Beginning of field
        - I-FIELD: Inside field (continuation)
        - O: Outside any field
        """
        # Get tokens
        tokens = self.processor.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0]
        )

        # Get softmax probabilities for confidence
        probs = torch.softmax(logits, dim=-1)

        fields = {}
        confidence_scores = {}
        current_field = None
        current_tokens = []
        current_confidences = []

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            label = self.FIELD_LABELS[pred.item()]
            prob = probs[i, pred.item()].item()

            if label.startswith("B-"):
                # Save previous field if exists
                if current_field and current_tokens:
                    field_name = current_field.replace("B-", "").replace("I-", "").lower()
                    fields[field_name] = self._merge_tokens(current_tokens)
                    confidence_scores[field_name] = np.mean(current_confidences)

                # Start new field
                current_field = label
                current_tokens = [token]
                current_confidences = [prob]

            elif label.startswith("I-") and current_field:
                # Continue current field
                current_tokens.append(token)
                current_confidences.append(prob)

            else:
                # Outside - save current field
                if current_field and current_tokens:
                    field_name = current_field.replace("B-", "").replace("I-", "").lower()
                    fields[field_name] = self._merge_tokens(current_tokens)
                    confidence_scores[field_name] = np.mean(current_confidences)

                current_field = None
                current_tokens = []
                current_confidences = []

        # Handle last field
        if current_field and current_tokens:
            field_name = current_field.replace("B-", "").replace("I-", "").lower()
            fields[field_name] = self._merge_tokens(current_tokens)
            confidence_scores[field_name] = np.mean(current_confidences)

        return fields, confidence_scores

    def _merge_tokens(self, tokens: List[str]) -> str:
        """Merge subword tokens into a single string."""
        text = " ".join(tokens)
        # Handle WordPiece tokens
        text = text.replace(" ##", "")
        text = text.replace("##", "")
        # Clean up special tokens
        text = text.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
        return text.strip()

    def fine_tune(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./layoutlm_finetuned",
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        use_lora: bool = True,
    ):
        """
        Fine-tune the model on custom data.

        For efficiency, uses LoRA (Low-Rank Adaptation) by default,
        which only trains ~0.1% of parameters.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Where to save fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            use_lora: Use LoRA for efficient fine-tuning
        """
        from transformers import TrainingArguments, Trainer

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType

                lora_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    r=16,  # Low-rank dimension
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["query", "value"],  # Attention layers
                )

                self.model = get_peft_model(self.model, lora_config)
                print(f"LoRA enabled. Trainable params: {self.model.print_trainable_parameters()}")

            except ImportError:
                print("PEFT not installed. Fine-tuning full model.")
                use_lora = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            logging_steps=50,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")


class LayoutLMv3FineTuned(LayoutLMv3Encoder):
    """
    LayoutLMv3 model fine-tuned specifically for invoice/PO extraction.

    Uses pre-trained checkpoint from HuggingFace Hub that has been
    fine-tuned on invoice datasets.
    """

    def __init__(
        self,
        model_name: str = "impira/layoutlm-invoices",  # Fine-tuned for invoices
        device: str = None,
    ):
        super().__init__(model_name=model_name, device=device)

        # Override labels for invoice-specific fields
        self.FIELD_LABELS = [
            "O",
            "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
            "B-VENDOR_NAME", "I-VENDOR_NAME",
            "B-INVOICE_DATE", "I-INVOICE_DATE",
            "B-DUE_DATE", "I-DUE_DATE",
            "B-TOTAL", "I-TOTAL",
            "B-TAX", "I-TAX",
            "B-SUBTOTAL", "I-SUBTOTAL",
            "B-LINE_ITEM", "I-LINE_ITEM",
        ]
