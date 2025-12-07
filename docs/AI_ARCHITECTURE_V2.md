# AI-Powered Document Understanding Architecture v2.0

## Senior AI Engineer Solution Design

**Author:** Akshay Karadkar
**Date:** December 2024

---

## Executive Summary

This document presents a **truly AI-powered** architecture for document understanding, moving beyond rule-based parsing to leverage state-of-the-art multi-modal transformers, vision-language models, and active learning systems.

---

## 1. Architecture Comparison

### v1.0 (Rule-Based) vs v2.0 (AI-Powered)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    v1.0 RULE-BASED APPROACH (Current)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PDF ──▶ pdfplumber ──▶ regex patterns ──▶ field extraction                │
│  Excel ──▶ pandas ──▶ column mapping ──▶ field extraction                  │
│  Word ──▶ python-docx ──▶ table iteration ──▶ field extraction             │
│  CSV ──▶ pandas ──▶ column aliases ──▶ field extraction                    │
│  Image ──▶ GPT-4o API ──▶ prompt ──▶ JSON parsing                          │
│                                                                              │
│  PROBLEMS:                                                                   │
│  • Each format needs custom code                                            │
│  • Brittle to format variations                                             │
│  • No learning from corrections                                             │
│  • Confidence is heuristic, not learned                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    v2.0 AI-POWERED APPROACH (New)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌─────────────────────────┐                              │
│  ANY DOCUMENT ───▶ │  UNIFIED VISUAL ENCODER │                              │
│  (PDF/Image/Scan)  │  (Document as Image)    │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                             │
│              ┌─────────────────┼─────────────────┐                          │
│              ▼                 ▼                 ▼                          │
│     ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                    │
│     │   Layout    │   │   Visual    │   │    Text     │                    │
│     │  Embeddings │   │  Features   │   │  Embeddings │                    │
│     └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                    │
│            │                 │                 │                            │
│            └─────────────────┼─────────────────┘                            │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │  TRI-MODAL FUSION   │                                  │
│                    │  (LayoutLMv3/Donut) │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ STRUCTURED OUTPUT   │                                  │
│                    │ (Token Classification│                                 │
│                    │  or Seq2Seq)        │                                  │
│                    └──────────┬──────────┘                                  │
│                               │                                             │
│                               ▼                                             │
│                    ┌─────────────────────┐                                  │
│                    │ CALIBRATED          │                                  │
│                    │ CONFIDENCE + HITL   │                                  │
│                    └─────────────────────┘                                  │
│                                                                              │
│  ADVANTAGES:                                                                │
│  • Single model handles ALL formats                                        │
│  • Learns document structure, not just text                                │
│  • Improves from human corrections (active learning)                       │
│  • Calibrated uncertainty quantification                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core AI Components

### 2.1 Document Understanding Models

| Model | Type | Use Case | Why |
|-------|------|----------|-----|
| **LayoutLMv3** | Tri-modal Transformer | Structured documents | Best accuracy on DocVQA (83.37 ANLS) |
| **Donut** | OCR-free Encoder-Decoder | End-to-end parsing | No OCR pipeline needed |
| **TrOCR** | Vision-to-Text Transformer | Handwriting | State-of-the-art HTR |
| **LLM (GPT-4/Claude)** | Vision-Language Model | Complex reasoning | Handles edge cases |

### 2.2 Model Selection Strategy

```python
class ModelRouter:
    """
    Intelligent routing based on document characteristics.
    This is AI-based routing, not file extension matching.
    """

    def __init__(self):
        self.layout_analyzer = LayoutAnalyzer()  # CNN-based
        self.text_density_model = TextDensityClassifier()
        self.handwriting_detector = HandwritingDetector()

    def route(self, document_image: np.ndarray) -> str:
        # Analyze document characteristics
        layout_complexity = self.layout_analyzer.predict(document_image)
        text_density = self.text_density_model.predict(document_image)
        has_handwriting = self.handwriting_detector.predict(document_image)

        if has_handwriting > 0.7:
            return "trocr_pipeline"  # Specialized for handwriting
        elif layout_complexity > 0.8:
            return "layoutlmv3_pipeline"  # Complex tables/forms
        elif text_density < 0.3:
            return "donut_pipeline"  # Sparse, image-heavy
        else:
            return "hybrid_pipeline"  # Combine approaches
```

---

## 3. Tri-Modal Document Encoder

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRI-MODAL DOCUMENT ENCODER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Document Image (any format rendered as image)                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         VISUAL BACKBONE                              │   │
│  │                    (ViT / Swin Transformer)                          │   │
│  │                                                                      │   │
│  │    Image ──▶ Patch Embeddings ──▶ [CLS] + Patch Tokens              │   │
│  │                     (16x16 patches)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          OCR ENGINE                                  │   │
│  │                   (TrOCR / PaddleOCR / API)                         │   │
│  │                                                                      │   │
│  │    Patches ──▶ Text Detection ──▶ Text Recognition ──▶ Words+BBoxes │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LAYOUT EMBEDDING                                │   │
│  │                                                                      │   │
│  │    For each word:                                                   │   │
│  │    • x_min, y_min, x_max, y_max (normalized 0-1000)                │   │
│  │    • Embedded via learned position embeddings                       │   │
│  │                                                                      │   │
│  │    Layout_Emb = Emb(x_min) + Emb(y_min) + Emb(x_max) + Emb(y_max)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TEXT EMBEDDING                                  │   │
│  │                   (WordPiece / BPE Tokenizer)                       │   │
│  │                                                                      │   │
│  │    Words ──▶ Subword Tokens ──▶ Token Embeddings                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TRI-MODAL FUSION                                │   │
│  │                                                                      │   │
│  │    Final_Embedding = Text_Emb + Layout_Emb + Visual_Emb            │   │
│  │                                                                      │   │
│  │    ┌─────────────────────────────────────────────────────────────┐ │   │
│  │    │           TRANSFORMER ENCODER (12 layers)                    │ │   │
│  │    │                                                              │ │   │
│  │    │   Multi-Head Self-Attention across all modalities           │ │   │
│  │    │   • Text tokens attend to visual patches                    │ │   │
│  │    │   • Visual patches attend to nearby text                    │ │   │
│  │    │   • Layout provides spatial context                         │ │   │
│  │    └─────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                    Contextualized Token Representations                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Matters for Senior AI Engineer Role

**Traditional Approach (Junior/Mid-level):**
```python
# Rule-based: Match column names
if "Order ID" in columns or "PO Number" in columns:
    order_id = row["Order ID"] or row["PO Number"]
```

**AI-Powered Approach (Senior-level):**
```python
# Model learns semantic meaning + spatial relationships
class DocumentFieldExtractor(nn.Module):
    def __init__(self):
        self.encoder = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=len(FIELD_LABELS)  # order_id, client_name, date, etc.
        )

    def forward(self, images, input_ids, attention_mask, bbox):
        # Model learns:
        # - "PO-2024-001" near top-right is likely order_id
        # - Numbers in rightmost column of a table are likely prices
        # - Text after "Ship to:" is likely delivery address
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=images
        )
        return outputs.logits  # Per-token field predictions
```

---

## 4. Structured Output Generation

### 4.1 Two Approaches

#### Approach A: Token Classification (LayoutLMv3)

```
Input:  [CLS] PO-2024-001 TechCorp Industries March 15, 2024 Widget $25.00 [SEP]
Labels: [O]   [ORDER_ID]  [CLIENT_NAME]       [DATE]         [ITEM]  [PRICE] [O]

Output: BIO tags per token → aggregate into fields
```

#### Approach B: Seq2Seq Generation (Donut)

```
Input:  Document Image
Output: <order_id>PO-2024-001</order_id><client>TechCorp</client>...

# Autoregressive generation of structured output
```

### 4.2 Hybrid Approach (Recommended)

```python
class HybridExtractor:
    """
    Combines strengths of both approaches:
    1. LayoutLMv3 for field detection (WHERE are the fields?)
    2. LLM for value extraction (WHAT are the values?)
    """

    def __init__(self):
        self.field_detector = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base-finetuned-invoice"
        )
        self.value_extractor = Claude()  # or GPT-4

    def extract(self, document_image):
        # Step 1: Detect field regions
        field_regions = self.field_detector.predict(document_image)
        # Output: {"order_id": BBox(x1,y1,x2,y2), "total": BBox(...), ...}

        # Step 2: Crop and extract values with LLM
        results = {}
        for field_name, bbox in field_regions.items():
            cropped = crop_region(document_image, bbox)
            value = self.value_extractor.extract_value(
                cropped,
                field_type=field_name
            )
            results[field_name] = value

        return results
```

---

## 5. Calibrated Confidence Scoring

### 5.1 The Problem with Naive Confidence

```python
# BAD: Rule-based confidence (what we had)
def naive_confidence(extraction):
    score = 0.0
    if extraction.order_id:
        score += 0.2
    if extraction.client_name:
        score += 0.2
    # ... arbitrary weights
    return score
```

### 5.2 AI-Powered Calibrated Confidence

```python
class CalibratedConfidenceScorer(nn.Module):
    """
    Learns to predict actual accuracy from model outputs.

    Key insight: Model softmax probabilities are often overconfident.
    We need to CALIBRATE them to match empirical accuracy.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Calibration network (learned from validation data)
        self.calibrator = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Temperature scaling (simple but effective)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        # Get base model logits
        logits = self.base_model(x)

        # Apply temperature scaling
        calibrated_logits = logits / self.temperature

        # Get calibrated probabilities
        probs = F.softmax(calibrated_logits, dim=-1)

        # Predict actual accuracy (not just max prob)
        hidden = self.base_model.get_hidden_states(x)
        confidence = self.calibrator(hidden)

        return probs, confidence

    def calibration_loss(self, predicted_conf, actual_accuracy):
        """
        Train calibrator so predicted confidence matches actual accuracy.

        If model says 80% confident, it should be correct 80% of the time.
        """
        return F.mse_loss(predicted_conf, actual_accuracy)
```

### 5.3 Uncertainty Quantification with MC Dropout

```python
class UncertaintyEstimator:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.

    Key insight: Run inference multiple times with dropout enabled.
    High variance = model is uncertain = send to human review.
    """

    def __init__(self, model, n_samples=10):
        self.model = model
        self.n_samples = n_samples

    def predict_with_uncertainty(self, x):
        self.model.train()  # Enable dropout

        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Epistemic uncertainty (model uncertainty)
        epistemic = predictions.var(dim=0)

        # Aleatoric uncertainty (data uncertainty) - from softmax entropy
        aleatoric = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)

        # Total uncertainty
        total_uncertainty = epistemic.mean() + aleatoric.mean()

        # Convert to confidence
        confidence = 1.0 / (1.0 + total_uncertainty)

        return mean_pred, confidence, {
            "epistemic": epistemic,
            "aleatoric": aleatoric
        }
```

---

## 6. Active Learning Pipeline

### 6.1 Why Active Learning Matters

```
Traditional Approach:
├── Deploy model
├── Collect errors manually
├── Retrain quarterly
└── Hope it gets better

Active Learning Approach:
├── Deploy model with uncertainty estimation
├── Route uncertain samples to human review
├── Collect labels on MOST INFORMATIVE samples
├── Continuously retrain on hard cases
└── Model improves on exact failure modes
```

### 6.2 Implementation

```python
class ActiveLearningPipeline:
    """
    Continuously improves model by learning from human corrections.
    """

    def __init__(self, model, uncertainty_threshold=0.7):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.correction_buffer = []
        self.min_samples_for_retrain = 100

    def process_document(self, document):
        # Get prediction with uncertainty
        prediction, confidence, uncertainty_breakdown = \
            self.model.predict_with_uncertainty(document)

        result = {
            "prediction": prediction,
            "confidence": confidence,
            "needs_review": confidence < self.uncertainty_threshold
        }

        if result["needs_review"]:
            # Queue for human review
            self.queue_for_review(document, prediction, uncertainty_breakdown)

        return result

    def receive_human_correction(self, document_id, corrected_labels):
        """Called when human reviews and corrects a prediction."""

        # Store correction
        self.correction_buffer.append({
            "document_id": document_id,
            "corrected_labels": corrected_labels,
            "timestamp": datetime.now()
        })

        # Trigger retraining if enough corrections collected
        if len(self.correction_buffer) >= self.min_samples_for_retrain:
            self.trigger_fine_tuning()

    def trigger_fine_tuning(self):
        """Fine-tune model on collected corrections using LoRA."""

        from peft import LoraConfig, get_peft_model

        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1
        )

        # Create PEFT model
        peft_model = get_peft_model(self.model, lora_config)

        # Fine-tune on corrections
        trainer = Trainer(
            model=peft_model,
            train_dataset=self.corrections_to_dataset(),
            args=TrainingArguments(
                num_train_epochs=3,
                per_device_train_batch_size=4,
                learning_rate=2e-5,
            )
        )
        trainer.train()

        # Update production model
        self.model = peft_model
        self.correction_buffer = []

        logger.info(f"Model fine-tuned on {len(self.correction_buffer)} corrections")
```

### 6.3 Uncertainty Sampling Strategy

```python
class UncertaintySampler:
    """
    Select most informative samples for human labeling.
    """

    @staticmethod
    def select_for_review(predictions, budget=10):
        """
        Select samples that will most improve the model.

        Strategies:
        1. Least confidence: lowest max probability
        2. Margin sampling: smallest margin between top 2 predictions
        3. Entropy sampling: highest prediction entropy
        """

        # Calculate uncertainty scores
        uncertainties = []
        for pred in predictions:
            probs = pred["probabilities"]

            # Least confidence
            least_conf = 1 - np.max(probs)

            # Margin (difference between top 2)
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1]

            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Combined score (can be weighted)
            combined = least_conf + (1 - margin) + entropy
            uncertainties.append(combined)

        # Select top-k most uncertain
        indices = np.argsort(uncertainties)[::-1][:budget]

        return indices
```

---

## 7. Production Architecture

### 7.1 System Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI-POWERED DOCUMENT PROCESSING SYSTEM                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         INGESTION LAYER                               │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │  │   PDF   │  │  Image  │  │  Email  │  │   API   │  │   S3    │   │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │  │
│  │       └───────────────────┬───────────────────┴───────────┘         │  │
│  └───────────────────────────┼──────────────────────────────────────────┘  │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      PREPROCESSING LAYER                              │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │  PDF→Image  │  │   Deskew    │  │   Denoise   │                  │  │
│  │  │  Renderer   │  │  Correction │  │   Filter    │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  └───────────────────────────┼──────────────────────────────────────────┘  │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      AI PROCESSING LAYER                              │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐│  │
│  │  │                    MODEL ROUTER (AI-based)                      ││  │
│  │  │  Analyzes: layout complexity, text density, handwriting presence││  │
│  │  └─────────────────────────────┬───────────────────────────────────┘│  │
│  │                                │                                     │  │
│  │        ┌───────────────────────┼───────────────────────┐            │  │
│  │        ▼                       ▼                       ▼            │  │
│  │  ┌───────────┐          ┌───────────┐          ┌───────────┐       │  │
│  │  │LayoutLMv3 │          │   Donut   │          │   TrOCR   │       │  │
│  │  │ (Tables/  │          │  (Sparse  │          │(Handwriting)│      │  │
│  │  │  Forms)   │          │   Docs)   │          │           │       │  │
│  │  └─────┬─────┘          └─────┬─────┘          └─────┬─────┘       │  │
│  │        │                      │                      │              │  │
│  │        └──────────────────────┼──────────────────────┘              │  │
│  │                               ▼                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐│  │
│  │  │                    ENSEMBLE / FUSION                            ││  │
│  │  │  Combines outputs from multiple models with learned weights     ││  │
│  │  └─────────────────────────────────────────────────────────────────┘│  │
│  │                               │                                      │  │
│  │                               ▼                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐│  │
│  │  │                 LLM REASONING LAYER (GPT-4/Claude)              ││  │
│  │  │  • Resolves ambiguities                                         ││  │
│  │  │  • Validates extracted values                                   ││  │
│  │  │  • Handles edge cases                                           ││  │
│  │  └─────────────────────────────────────────────────────────────────┘│  │
│  └───────────────────────────────┼──────────────────────────────────────┘  │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    CONFIDENCE & ROUTING LAYER                         │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │  Calibrated │  │   MC       │  │  Uncertainty │                  │  │
│  │  │  Confidence │  │  Dropout   │  │   Routing    │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  │                           │                                          │  │
│  │         ┌─────────────────┼─────────────────┐                       │  │
│  │         ▼                 ▼                 ▼                       │  │
│  │    [HIGH CONF]      [MEDIUM CONF]     [LOW CONF]                   │  │
│  │    Auto-approve     Sample review     Human review                  │  │
│  └───────────────────────────┼──────────────────────────────────────────┘  │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    OUTPUT & LEARNING LAYER                            │  │
│  │                                                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │  JSON       │  │   Human     │  │   Active    │                  │  │
│  │  │  Export     │  │   Review UI │  │   Learning  │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  │                           │                                          │  │
│  │                           ▼                                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐│  │
│  │  │                    FEEDBACK LOOP                                ││  │
│  │  │  Human corrections → LoRA fine-tuning → Model improvement      ││  │
│  │  └─────────────────────────────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Differentiators for Senior AI Engineer

### 8.1 What This Architecture Demonstrates

| Skill | Demonstration |
|-------|---------------|
| **Deep Learning** | Custom LayoutLMv3 fine-tuning, multi-modal fusion |
| **MLOps** | Active learning pipeline, continuous improvement |
| **Uncertainty Quantification** | MC Dropout, calibrated confidence |
| **System Design** | Scalable production architecture |
| **Research Awareness** | State-of-the-art models (2024-2025) |
| **Engineering** | LoRA fine-tuning, efficient inference |

### 8.2 Technical Depth

```python
# This is what a Senior AI Engineer writes:

class DocumentUnderstandingSystem:
    """
    Production-grade document understanding with:
    - Multi-modal transformers (LayoutLMv3, Donut)
    - Calibrated uncertainty estimation
    - Active learning from human feedback
    - Efficient fine-tuning with LoRA
    """

    def __init__(self, config: SystemConfig):
        # Model ensemble
        self.layout_model = LayoutLMv3.from_pretrained(config.layout_model)
        self.ocr_free_model = Donut.from_pretrained(config.donut_model)
        self.handwriting_model = TrOCR.from_pretrained(config.trocr_model)

        # Confidence calibration
        self.calibrator = TemperatureScaling(config.calibration_temp)
        self.uncertainty_estimator = MCDropoutEstimator(n_samples=10)

        # Active learning
        self.active_learner = ActiveLearningPipeline(
            uncertainty_threshold=config.uncertainty_threshold,
            min_samples_for_retrain=config.retrain_threshold
        )

        # Model router (learned, not rule-based)
        self.router = ModelRouter.from_pretrained(config.router_model)

    def process(self, document: Document) -> ExtractionResult:
        # 1. Preprocess
        image = self.preprocess(document)

        # 2. Route to appropriate model(s)
        route = self.router.predict(image)

        # 3. Extract with selected model(s)
        if route == "ensemble":
            predictions = self.ensemble_predict(image)
        else:
            predictions = self.single_model_predict(image, route)

        # 4. Calibrate confidence
        calibrated_conf = self.calibrator(predictions.logits)

        # 5. Estimate uncertainty
        uncertainty = self.uncertainty_estimator(image)

        # 6. Route based on confidence
        result = ExtractionResult(
            fields=predictions.fields,
            confidence=calibrated_conf,
            uncertainty=uncertainty,
            needs_review=calibrated_conf < self.config.review_threshold
        )

        # 7. Queue for active learning if needed
        if result.needs_review:
            self.active_learner.queue(document, result)

        return result
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation
- [ ] Set up LayoutLMv3 and Donut models
- [ ] Implement document-to-image preprocessing
- [ ] Create unified extraction interface

### Phase 2: Intelligence
- [ ] Train model router on document characteristics
- [ ] Implement ensemble fusion layer
- [ ] Add LLM reasoning for edge cases

### Phase 3: Confidence
- [ ] Implement MC Dropout uncertainty
- [ ] Train calibration network
- [ ] Build confidence-based routing

### Phase 4: Learning
- [ ] Build human review interface
- [ ] Implement LoRA fine-tuning pipeline
- [ ] Create active learning feedback loop

### Phase 5: Production
- [ ] Deploy with vLLM/Triton
- [ ] Set up monitoring and alerting
- [ ] Document and test thoroughly

---

## 10. Conclusion

This architecture demonstrates the skills expected of a **Senior AI Engineer**:

1. **Not just calling APIs** - Building custom models and training pipelines
2. **Understanding trade-offs** - When to use LayoutLMv3 vs Donut vs LLM
3. **Production thinking** - Confidence calibration, active learning, scaling
4. **Research awareness** - Using 2024-2025 state-of-the-art approaches
5. **System design** - End-to-end architecture with feedback loops

The key insight: **A truly AI-powered system learns and improves**, it doesn't just execute fixed rules.
