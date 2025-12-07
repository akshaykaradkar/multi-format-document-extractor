"""
Active Learning Pipeline

Active learning enables continuous model improvement by:
1. Identifying uncertain/informative samples
2. Routing them for human annotation
3. Fine-tuning on the new labels
4. Deploying improved model

This is key for production ML systems - the model improves
on its exact failure modes over time.

Key strategies:
- Uncertainty sampling: Select samples with lowest confidence
- Query-by-committee: Select samples where models disagree
- Expected model change: Select samples that would most change the model
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from collections import deque

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


@dataclass
class LabeledSample:
    """A sample with human-provided labels."""
    sample_id: str
    document_path: str
    predicted_fields: Dict[str, str]
    corrected_fields: Dict[str, str]
    prediction_confidence: float
    timestamp: datetime
    annotator_id: Optional[str] = None
    annotation_time_seconds: Optional[float] = None


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning pipeline."""
    uncertainty_threshold: float = 0.7
    min_samples_for_retrain: int = 50
    max_buffer_size: int = 1000
    sampling_strategy: str = "uncertainty"  # or "diversity", "hybrid"
    retrain_epochs: int = 3
    use_lora: bool = True
    lora_rank: int = 16
    checkpoint_dir: str = "./checkpoints"


class UncertaintySampler:
    """
    Selects most informative samples for human annotation.

    Strategies:
    1. Least confidence: lowest max probability
    2. Margin sampling: smallest margin between top 2
    3. Entropy sampling: highest prediction entropy
    4. Combined: weighted combination
    """

    def __init__(self, strategy: str = "combined"):
        self.strategy = strategy

    def score_sample(
        self,
        probabilities: np.ndarray,
    ) -> float:
        """
        Score a sample's informativeness.

        Higher score = more informative = should annotate.
        """
        if self.strategy == "least_confidence":
            return self._least_confidence(probabilities)
        elif self.strategy == "margin":
            return self._margin_sampling(probabilities)
        elif self.strategy == "entropy":
            return self._entropy_sampling(probabilities)
        elif self.strategy == "combined":
            return self._combined_sampling(probabilities)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _least_confidence(self, probs: np.ndarray) -> float:
        """1 - max probability."""
        return 1.0 - np.max(probs)

    def _margin_sampling(self, probs: np.ndarray) -> float:
        """1 - (top1 - top2). Small margin = uncertain."""
        sorted_probs = np.sort(probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        return 1.0 - margin

    def _entropy_sampling(self, probs: np.ndarray) -> float:
        """Entropy of distribution. Higher = more uncertain."""
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        # Normalize by max entropy
        max_entropy = np.log(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0

    def _combined_sampling(self, probs: np.ndarray) -> float:
        """Weighted combination of strategies."""
        lc = self._least_confidence(probs)
        margin = self._margin_sampling(probs)
        entropy = self._entropy_sampling(probs)
        return 0.4 * lc + 0.3 * margin + 0.3 * entropy

    def select_samples(
        self,
        samples: List[Dict],
        budget: int,
    ) -> List[int]:
        """
        Select top-k most informative samples.

        Args:
            samples: List of dicts with 'probabilities' key
            budget: How many samples to select

        Returns:
            Indices of selected samples
        """
        scores = []
        for sample in samples:
            probs = np.array(sample['probabilities'])
            score = self.score_sample(probs)
            scores.append(score)

        # Return indices of top-k scores
        indices = np.argsort(scores)[::-1][:budget]
        return indices.tolist()


class DiversitySampler:
    """
    Selects diverse samples to maximize coverage.

    Uses clustering or embedding distance to ensure
    selected samples are different from each other.
    """

    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters

    def select_samples(
        self,
        embeddings: np.ndarray,
        budget: int,
    ) -> List[int]:
        """
        Select diverse samples using k-means clustering.

        Args:
            embeddings: Document embeddings (N, D)
            budget: How many samples to select

        Returns:
            Indices of selected samples
        """
        from sklearn.cluster import KMeans

        # Cluster embeddings
        n_clusters = min(self.n_clusters, budget, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select sample closest to each cluster center
        selected = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Find closest to centroid
                cluster_embeddings = embeddings[cluster_indices]
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected.append(closest_idx)

        # If need more, sample from remaining
        remaining_budget = budget - len(selected)
        if remaining_budget > 0:
            remaining = [i for i in range(len(embeddings)) if i not in selected]
            additional = np.random.choice(
                remaining,
                size=min(remaining_budget, len(remaining)),
                replace=False
            )
            selected.extend(additional.tolist())

        return selected[:budget]


class ActiveLearningPipeline:
    """
    Complete active learning pipeline for document understanding.

    Workflow:
    1. Model makes prediction with uncertainty
    2. Low-confidence samples queued for review
    3. Humans correct predictions
    4. Model fine-tuned on corrections
    5. Improved model deployed

    This creates a feedback loop that continuously improves
    the model on its actual failure modes.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ActiveLearningConfig = None,
    ):
        """
        Initialize active learning pipeline.

        Args:
            model: Base model to improve
            config: Pipeline configuration
        """
        self.model = model
        self.config = config or ActiveLearningConfig()

        # Sample buffer
        self.correction_buffer: deque = deque(maxlen=self.config.max_buffer_size)
        self.pending_review: Dict[str, Dict] = {}

        # Samplers
        self.uncertainty_sampler = UncertaintySampler(
            strategy=self.config.sampling_strategy
        )

        # Training history
        self.training_history: List[Dict] = []

        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def process_prediction(
        self,
        sample_id: str,
        document_path: str,
        prediction: Dict[str, str],
        confidence: float,
        probabilities: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Process a model prediction and determine if review needed.

        Args:
            sample_id: Unique identifier for this sample
            document_path: Path to document
            prediction: Predicted fields
            confidence: Overall confidence score
            probabilities: Optional per-class probabilities

        Returns:
            Dict with prediction and review status
        """
        needs_review = confidence < self.config.uncertainty_threshold

        result = {
            'sample_id': sample_id,
            'prediction': prediction,
            'confidence': confidence,
            'needs_review': needs_review,
        }

        if needs_review:
            # Add to review queue
            self.pending_review[sample_id] = {
                'document_path': document_path,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'timestamp': datetime.now().isoformat(),
            }
            result['review_priority'] = self._calculate_priority(confidence, probabilities)

        return result

    def _calculate_priority(
        self,
        confidence: float,
        probabilities: Optional[np.ndarray],
    ) -> str:
        """Calculate review priority based on uncertainty."""
        if confidence < 0.5:
            return 'high'
        elif confidence < 0.7:
            return 'medium'
        else:
            return 'low'

    def receive_correction(
        self,
        sample_id: str,
        corrected_fields: Dict[str, str],
        annotator_id: Optional[str] = None,
        annotation_time: Optional[float] = None,
    ) -> Dict:
        """
        Receive human correction for a prediction.

        Args:
            sample_id: Sample identifier
            corrected_fields: Human-corrected field values
            annotator_id: Who made the correction
            annotation_time: How long annotation took

        Returns:
            Status dict
        """
        if sample_id not in self.pending_review:
            return {'status': 'error', 'message': 'Sample not found in review queue'}

        # Get original prediction
        original = self.pending_review[sample_id]

        # Create labeled sample
        labeled_sample = LabeledSample(
            sample_id=sample_id,
            document_path=original['document_path'],
            predicted_fields=original['prediction'],
            corrected_fields=corrected_fields,
            prediction_confidence=original['confidence'],
            timestamp=datetime.now(),
            annotator_id=annotator_id,
            annotation_time_seconds=annotation_time,
        )

        # Add to correction buffer
        self.correction_buffer.append(labeled_sample)

        # Remove from pending
        del self.pending_review[sample_id]

        # Check if should trigger retraining
        should_retrain = len(self.correction_buffer) >= self.config.min_samples_for_retrain

        return {
            'status': 'success',
            'buffer_size': len(self.correction_buffer),
            'should_retrain': should_retrain,
        }

    def get_samples_for_annotation(
        self,
        budget: int = 10,
    ) -> List[Dict]:
        """
        Get prioritized samples for human annotation.

        Uses uncertainty sampling to select most informative samples.

        Args:
            budget: How many samples to return

        Returns:
            List of samples needing annotation
        """
        if not self.pending_review:
            return []

        # Convert to list for sampling
        samples = list(self.pending_review.values())
        sample_ids = list(self.pending_review.keys())

        # If we have probabilities, use uncertainty sampling
        if samples[0].get('probabilities'):
            indices = self.uncertainty_sampler.select_samples(
                [{'probabilities': s['probabilities']} for s in samples],
                budget
            )
        else:
            # Fall back to lowest confidence
            confidences = [s['confidence'] for s in samples]
            indices = np.argsort(confidences)[:budget].tolist()

        # Return selected samples with IDs
        selected = []
        for idx in indices:
            sample = samples[idx].copy()
            sample['sample_id'] = sample_ids[idx]
            selected.append(sample)

        return selected

    def trigger_fine_tuning(self) -> Dict:
        """
        Fine-tune model on collected corrections.

        Uses LoRA for efficient fine-tuning.

        Returns:
            Training results
        """
        if len(self.correction_buffer) < self.config.min_samples_for_retrain:
            return {
                'status': 'skipped',
                'reason': f'Not enough samples ({len(self.correction_buffer)}/{self.config.min_samples_for_retrain})'
            }

        print(f"Starting fine-tuning on {len(self.correction_buffer)} samples...")

        # Prepare training data
        train_data = self._prepare_training_data()

        # Apply LoRA if available and configured
        if self.config.use_lora and HAS_PEFT:
            self.model = self._apply_lora(self.model)

        # Fine-tune
        results = self._fine_tune(train_data)

        # Save checkpoint
        checkpoint_path = self._save_checkpoint()

        # Clear buffer
        processed_count = len(self.correction_buffer)
        self.correction_buffer.clear()

        # Log training
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples_used': processed_count,
            'results': results,
            'checkpoint': checkpoint_path,
        })

        return {
            'status': 'success',
            'samples_processed': processed_count,
            'results': results,
            'checkpoint': checkpoint_path,
        }

    def _prepare_training_data(self) -> List[Dict]:
        """Convert corrections to training format."""
        train_data = []

        for sample in self.correction_buffer:
            train_data.append({
                'document_path': sample.document_path,
                'labels': sample.corrected_fields,
            })

        return train_data

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """
        Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning.

        LoRA only trains ~0.1% of parameters while achieving
        95% of full fine-tuning performance.
        """
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=self.config.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # Attention layers
        )

        peft_model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"LoRA enabled: {trainable_params:,} / {total_params:,} trainable ({100 * trainable_params / total_params:.2f}%)")

        return peft_model

    def _fine_tune(self, train_data: List[Dict]) -> Dict:
        """
        Fine-tune the model on training data.

        This is a simplified version - production would use
        proper DataLoader, validation, early stopping, etc.
        """
        from torch.optim import AdamW

        self.model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-5
        )

        total_loss = 0
        num_batches = 0

        for epoch in range(self.config.retrain_epochs):
            epoch_loss = 0

            for sample in train_data:
                # In production, this would properly prepare inputs
                # For now, simulate training step
                optimizer.zero_grad()

                # Placeholder loss calculation
                # Real implementation would:
                # 1. Load document image
                # 2. Run through model
                # 3. Calculate loss against corrected labels
                loss = torch.tensor(0.1, requires_grad=True)  # Placeholder

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_data):.4f}")
            total_loss += epoch_loss

        self.model.eval()

        return {
            'epochs': self.config.retrain_epochs,
            'avg_loss': total_loss / num_batches if num_batches > 0 else 0,
            'samples': len(train_data),
        }

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = Path(self.config.checkpoint_dir) / f"model_{timestamp}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
        }, checkpoint_path)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'pending_review': len(self.pending_review),
            'correction_buffer': len(self.correction_buffer),
            'total_training_rounds': len(self.training_history),
            'total_samples_trained': sum(
                h.get('samples_used', 0) for h in self.training_history
            ),
        }

    def export_corrections(self, output_path: str):
        """Export corrections for analysis or external training."""
        corrections = []

        for sample in self.correction_buffer:
            corrections.append({
                'sample_id': sample.sample_id,
                'document_path': sample.document_path,
                'predicted': sample.predicted_fields,
                'corrected': sample.corrected_fields,
                'confidence': sample.prediction_confidence,
                'timestamp': sample.timestamp.isoformat(),
            })

        with open(output_path, 'w') as f:
            json.dump(corrections, f, indent=2)

        print(f"Exported {len(corrections)} corrections to {output_path}")
