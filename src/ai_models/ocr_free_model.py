"""
Donut: OCR-Free Document Understanding Transformer

Donut (Document Understanding Transformer) is an end-to-end model that
reads documents directly from pixels WITHOUT requiring a separate OCR step.

Key advantages:
1. No OCR error propagation
2. Faster inference (single forward pass)
3. Better for noisy/degraded documents
4. Learns to read any text style

Architecture:
- Encoder: Swin Transformer (visual)
- Decoder: BART-style transformer (text generation)

Reference: https://arxiv.org/abs/2111.15664
"""

import torch
import torch.nn as nn
from PIL import Image
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from transformers import (
        DonutProcessor,
        VisionEncoderDecoderModel,
        GenerationConfig,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class DonutExtractionResult:
    """Result from Donut extraction."""
    raw_output: str
    parsed_fields: Dict[str, any]
    confidence: float
    generation_scores: Optional[List[float]] = None


class DonutExtractor:
    """
    OCR-free document extractor using Donut model.

    This model generates structured output directly from document images
    without requiring a separate OCR step. It's particularly effective for:
    - Noisy or degraded documents
    - Non-standard layouts
    - Documents where OCR errors are problematic

    The model is trained to generate JSON/XML structured output
    autoregressively from image input.
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa",
        device: str = None,
        max_length: int = 512,
    ):
        """
        Initialize Donut extractor.

        Args:
            model_name: HuggingFace model identifier
                Options:
                - "naver-clova-ix/donut-base" (base model)
                - "naver-clova-ix/donut-base-finetuned-docvqa" (DocVQA)
                - "naver-clova-ix/donut-base-finetuned-cord-v2" (receipts)
            device: Device to run on
            max_length: Maximum generation length
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers"
            )

        self.device = device or self._get_device()
        self.max_length = max_length
        self.model_name = model_name

        self._init_model()

    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_model(self):
        """Initialize Donut model and processor."""
        print(f"Loading Donut from {self.model_name}...")

        self.processor = DonutProcessor.from_pretrained(self.model_name)

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"Donut loaded on {self.device}")

    def preprocess_image(self, image: Union[Image.Image, Path, str]) -> Image.Image:
        """Load and preprocess document image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def extract(
        self,
        image: Union[Image.Image, Path, str],
        task_prompt: str = "<s_docvqa>",
        question: Optional[str] = None,
    ) -> DonutExtractionResult:
        """
        Extract information from document image.

        Args:
            image: Document image
            task_prompt: Task-specific prompt
                - "<s_docvqa>" for document QA
                - "<s_cord-v2>" for receipt parsing
                - Custom prompt for custom tasks
            question: Question for DocVQA mode

        Returns:
            DonutExtractionResult with extracted data
        """
        image = self.preprocess_image(image)

        # Prepare decoder prompt
        if question:
            # DocVQA mode
            decoder_prompt = f"{task_prompt}<s_question>{question}</s_question><s_answer>"
        else:
            # Direct extraction mode
            decoder_prompt = task_prompt

        # Process image
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # Prepare decoder input
        decoder_input_ids = self.processor.tokenizer(
            decoder_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=4,  # Beam search for better quality
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode output
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

        # Parse structured output
        parsed_fields = self._parse_output(sequence)

        # Calculate confidence from generation scores
        confidence = self._calculate_confidence(outputs)

        return DonutExtractionResult(
            raw_output=sequence,
            parsed_fields=parsed_fields,
            confidence=confidence,
            generation_scores=None,  # Can extract if needed
        )

    def extract_purchase_order(
        self,
        image: Union[Image.Image, Path, str],
    ) -> DonutExtractionResult:
        """
        Extract purchase order fields with custom prompt.

        Uses a structured prompt to guide extraction of PO-specific fields.
        """
        # Custom prompt for PO extraction
        extraction_prompt = """<s_purchaseorder>
Extract the following fields:
- order_id: The purchase order number
- client_name: The company name
- order_date: The order date (YYYY-MM-DD)
- delivery_date: The delivery/ship date (YYYY-MM-DD)
- items: List of line items with product_code, description, quantity, unit_price, total_price
- order_total: The total amount
- special_instructions: Any special notes
</s_purchaseorder>"""

        return self.extract(image, task_prompt=extraction_prompt)

    def _parse_output(self, output: str) -> Dict[str, any]:
        """
        Parse Donut output into structured fields.

        Donut typically generates XML-like or JSON output.
        """
        # Try JSON parsing first
        try:
            # Find JSON in output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Try XML-like parsing
        fields = {}
        patterns = [
            (r'<order_id>(.*?)</order_id>', 'order_id'),
            (r'<client_name>(.*?)</client_name>', 'client_name'),
            (r'<order_date>(.*?)</order_date>', 'order_date'),
            (r'<delivery_date>(.*?)</delivery_date>', 'delivery_date'),
            (r'<total>(.*?)</total>', 'order_total'),
            (r'<answer>(.*?)</answer>', 'answer'),  # DocVQA format
        ]

        for pattern, field_name in patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                fields[field_name] = match.group(1).strip()

        # If no structured parsing worked, return raw
        if not fields:
            fields['raw_text'] = output

        return fields

    def _calculate_confidence(self, outputs) -> float:
        """
        Calculate confidence score from generation.

        Uses average token probability as confidence measure.
        """
        if outputs.scores is None:
            return 0.5  # Default confidence

        # Get probabilities for generated tokens
        probs = []
        for score in outputs.scores:
            # Softmax to get probabilities
            token_probs = torch.softmax(score, dim=-1)
            # Get probability of selected token
            max_prob = token_probs.max(dim=-1).values
            probs.append(max_prob.mean().item())

        if probs:
            return sum(probs) / len(probs)
        return 0.5

    def fine_tune_for_po(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./donut_finetuned_po",
        num_epochs: int = 3,
    ):
        """
        Fine-tune Donut for purchase order extraction.

        This creates a custom model that generates PO-specific
        structured output from document images.

        Args:
            train_dataset: Dataset of (image, target_json) pairs
            eval_dataset: Optional evaluation dataset
            output_dir: Where to save fine-tuned model
            num_epochs: Training epochs
        """
        from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,  # Donut needs more memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),  # Mixed precision
            logging_steps=50,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        print(f"Fine-tuned Donut saved to {output_dir}")


class DonutReceiptParser(DonutExtractor):
    """
    Donut model fine-tuned for receipt/invoice parsing.

    Uses the CORD (Consolidated Receipt Dataset) fine-tuned checkpoint.
    """

    def __init__(self, device: str = None):
        super().__init__(
            model_name="naver-clova-ix/donut-base-finetuned-cord-v2",
            device=device,
        )

    def extract_receipt(
        self,
        image: Union[Image.Image, Path, str],
    ) -> DonutExtractionResult:
        """Extract receipt fields."""
        return self.extract(image, task_prompt="<s_cord-v2>")
