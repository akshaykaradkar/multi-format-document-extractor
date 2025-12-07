"""
Model Ensemble for Document Understanding

Combines predictions from multiple models for improved accuracy
and robustness. Uses learned weights to fuse outputs.

Ensemble strategies:
1. Simple averaging
2. Weighted averaging (learned weights)
3. Stacking (meta-learner)
4. Confidence-based selection
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class FusionStrategy(Enum):
    """Available fusion strategies."""
    AVERAGE = "average"
    WEIGHTED = "weighted"
    STACKING = "stacking"
    CONFIDENCE = "confidence"
    VOTING = "voting"


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    fields: Dict[str, str]
    confidence_scores: Dict[str, float]
    model_contributions: Dict[str, float]
    individual_predictions: Dict[str, Dict]


class ModelEnsemble:
    """
    Ensemble of document understanding models.

    Combines:
    - LayoutLMv3 (layout-aware)
    - Donut (OCR-free)
    - TrOCR (handwriting)
    - LLM (reasoning)

    The ensemble learns optimal weights for each model
    based on document characteristics.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED,
        device: str = None,
    ):
        """
        Initialize ensemble.

        Args:
            models: Dict of model_name -> model instance
            fusion_strategy: How to combine predictions
            device: Device to run on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = models or {}
        self.fusion_strategy = fusion_strategy

        # Learned weights for each model
        if models:
            self.model_weights = nn.Parameter(
                torch.ones(len(models)) / len(models)
            )
        else:
            self.model_weights = None

        # Meta-learner for stacking
        if fusion_strategy == FusionStrategy.STACKING:
            self._init_meta_learner()

    def _init_meta_learner(self):
        """Initialize meta-learner for stacking ensemble."""
        # Number of fields * number of models -> final prediction
        input_dim = 10 * len(self.models)  # Approximate

        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Output fields
        )

    def add_model(self, name: str, model: nn.Module):
        """Add a model to the ensemble."""
        self.models[name] = model

        # Update weights
        n_models = len(self.models)
        self.model_weights = nn.Parameter(
            torch.ones(n_models) / n_models
        )

    def predict(
        self,
        image,
        document_characteristics: Dict = None,
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction.

        Args:
            image: Document image
            document_characteristics: From ModelRouter (optional)

        Returns:
            EnsemblePrediction with fused results
        """
        # Get predictions from each model
        individual_predictions = {}

        for name, model in self.models.items():
            try:
                # Each model has extract_fields method
                pred = model.extract_fields(image)
                individual_predictions[name] = {
                    'fields': pred.fields,
                    'confidence': pred.confidence_scores,
                }
            except Exception as e:
                print(f"Warning: Model {name} failed: {e}")
                individual_predictions[name] = None

        # Fuse predictions
        if self.fusion_strategy == FusionStrategy.AVERAGE:
            fused = self._fuse_average(individual_predictions)
        elif self.fusion_strategy == FusionStrategy.WEIGHTED:
            fused = self._fuse_weighted(individual_predictions, document_characteristics)
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE:
            fused = self._fuse_confidence(individual_predictions)
        elif self.fusion_strategy == FusionStrategy.VOTING:
            fused = self._fuse_voting(individual_predictions)
        else:
            fused = self._fuse_average(individual_predictions)

        return EnsemblePrediction(
            fields=fused['fields'],
            confidence_scores=fused['confidence'],
            model_contributions=fused['contributions'],
            individual_predictions=individual_predictions,
        )

    def _fuse_average(
        self,
        predictions: Dict[str, Dict],
    ) -> Dict:
        """Simple averaging of predictions."""
        all_fields = set()
        for pred in predictions.values():
            if pred:
                all_fields.update(pred['fields'].keys())

        fused_fields = {}
        fused_confidence = {}
        contributions = {name: 1.0 / len(predictions) for name in predictions}

        for field in all_fields:
            values = []
            confidences = []

            for name, pred in predictions.items():
                if pred and field in pred['fields']:
                    values.append(pred['fields'][field])
                    confidences.append(pred['confidence'].get(field, 0.5))

            if values:
                # For strings, use most common value
                fused_fields[field] = max(set(values), key=values.count)
                fused_confidence[field] = np.mean(confidences)

        return {
            'fields': fused_fields,
            'confidence': fused_confidence,
            'contributions': contributions,
        }

    def _fuse_weighted(
        self,
        predictions: Dict[str, Dict],
        characteristics: Dict = None,
    ) -> Dict:
        """
        Weighted averaging based on learned weights.

        Weights are adjusted based on document characteristics.
        """
        # Get base weights
        if self.model_weights is not None:
            weights = torch.softmax(self.model_weights, dim=0).detach().numpy()
        else:
            weights = np.ones(len(predictions)) / len(predictions)

        # Adjust weights based on characteristics
        if characteristics:
            weights = self._adjust_weights(weights, characteristics, predictions)

        # Normalize
        weights = weights / weights.sum()

        # Create weight mapping
        weight_map = dict(zip(predictions.keys(), weights))

        all_fields = set()
        for pred in predictions.values():
            if pred:
                all_fields.update(pred['fields'].keys())

        fused_fields = {}
        fused_confidence = {}

        for field in all_fields:
            weighted_values = {}

            for name, pred in predictions.items():
                if pred and field in pred['fields']:
                    value = pred['fields'][field]
                    conf = pred['confidence'].get(field, 0.5)
                    weight = weight_map[name] * conf

                    if value not in weighted_values:
                        weighted_values[value] = 0
                    weighted_values[value] += weight

            if weighted_values:
                # Select value with highest weighted vote
                fused_fields[field] = max(weighted_values, key=weighted_values.get)
                fused_confidence[field] = max(weighted_values.values())

        return {
            'fields': fused_fields,
            'confidence': fused_confidence,
            'contributions': weight_map,
        }

    def _fuse_confidence(
        self,
        predictions: Dict[str, Dict],
    ) -> Dict:
        """
        Select prediction from most confident model per field.

        Each field value comes from the model most confident about it.
        """
        all_fields = set()
        for pred in predictions.values():
            if pred:
                all_fields.update(pred['fields'].keys())

        fused_fields = {}
        fused_confidence = {}
        contributions = {name: 0.0 for name in predictions}

        for field in all_fields:
            best_value = None
            best_conf = -1
            best_model = None

            for name, pred in predictions.items():
                if pred and field in pred['fields']:
                    conf = pred['confidence'].get(field, 0.5)
                    if conf > best_conf:
                        best_conf = conf
                        best_value = pred['fields'][field]
                        best_model = name

            if best_value is not None:
                fused_fields[field] = best_value
                fused_confidence[field] = best_conf
                contributions[best_model] += 1

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return {
            'fields': fused_fields,
            'confidence': fused_confidence,
            'contributions': contributions,
        }

    def _fuse_voting(
        self,
        predictions: Dict[str, Dict],
    ) -> Dict:
        """
        Majority voting for each field.

        Each model votes for a value, most votes wins.
        """
        all_fields = set()
        for pred in predictions.values():
            if pred:
                all_fields.update(pred['fields'].keys())

        fused_fields = {}
        fused_confidence = {}
        contributions = {name: 0.0 for name in predictions}

        for field in all_fields:
            votes = {}

            for name, pred in predictions.items():
                if pred and field in pred['fields']:
                    value = pred['fields'][field]
                    if value not in votes:
                        votes[value] = []
                    votes[value].append(name)

            if votes:
                # Winner is value with most votes
                winner = max(votes, key=lambda v: len(votes[v]))
                fused_fields[field] = winner
                fused_confidence[field] = len(votes[winner]) / len(predictions)

                # Track contributions
                for name in votes[winner]:
                    contributions[name] += 1

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return {
            'fields': fused_fields,
            'confidence': fused_confidence,
            'contributions': contributions,
        }

    def _adjust_weights(
        self,
        base_weights: np.ndarray,
        characteristics: Dict,
        predictions: Dict,
    ) -> np.ndarray:
        """
        Adjust weights based on document characteristics.

        For example:
        - High handwriting → boost TrOCR weight
        - Complex layout → boost LayoutLMv3 weight
        - Noisy image → boost Donut weight
        """
        adjusted = base_weights.copy()
        model_names = list(predictions.keys())

        for i, name in enumerate(model_names):
            # Boost handwriting model for handwritten docs
            if 'trocr' in name.lower() and characteristics.get('handwriting_probability', 0) > 0.5:
                adjusted[i] *= 2.0

            # Boost layout model for complex layouts
            if 'layout' in name.lower() and characteristics.get('layout_complexity', 0) > 0.7:
                adjusted[i] *= 1.5

            # Boost OCR-free for noisy images
            if 'donut' in name.lower() and characteristics.get('noise_level', 0) > 0.5:
                adjusted[i] *= 1.5

        return adjusted

    def train_weights(
        self,
        train_data: List[tuple],
        epochs: int = 10,
        lr: float = 0.01,
    ):
        """
        Learn optimal ensemble weights from training data.

        Args:
            train_data: List of (image, ground_truth_fields) tuples
            epochs: Training epochs
            lr: Learning rate
        """
        if self.model_weights is None:
            return

        optimizer = torch.optim.Adam([self.model_weights], lr=lr)

        for epoch in range(epochs):
            total_loss = 0

            for image, ground_truth in train_data:
                # Get predictions
                pred = self.predict(image)

                # Calculate loss (field matching)
                loss = self._calculate_loss(pred.fields, ground_truth)
                total_loss += loss.item()

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_data):.4f}")

    def _calculate_loss(
        self,
        predicted: Dict[str, str],
        ground_truth: Dict[str, str],
    ) -> torch.Tensor:
        """Calculate loss for field predictions."""
        # Simple matching loss
        correct = 0
        total = len(ground_truth)

        for field, gt_value in ground_truth.items():
            if field in predicted and predicted[field] == gt_value:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        loss = torch.tensor(1.0 - accuracy, requires_grad=True)

        return loss
