"""
Calibrated Confidence Scoring

Unlike naive confidence (softmax probabilities), calibrated confidence
accurately reflects the probability of being correct.

Key insight: Model softmax probabilities are often overconfident.
A model saying "90% confident" might only be correct 70% of the time.

This module implements:
1. Temperature Scaling - Simple post-hoc calibration
2. Platt Scaling - Logistic regression on logits
3. MC Dropout - Bayesian uncertainty estimation
4. Ensemble Uncertainty - Disagreement between models

Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import optimize
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationResult:
    """Result from calibration analysis."""
    calibrated_confidence: float
    raw_confidence: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    reliability_diagram: Optional[Dict] = None


@dataclass
class UncertaintyBreakdown:
    """Detailed uncertainty breakdown."""
    epistemic: float  # Reducible with more training data
    aleatoric: float  # Irreducible noise in data
    total: float
    needs_review: bool
    reason: str


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for confidence calibration.

    The simplest and most effective calibration method.
    Learns a single temperature parameter T to scale logits:

    calibrated_probs = softmax(logits / T)

    T > 1: Reduces confidence (softens distribution)
    T < 1: Increases confidence (sharpens distribution)
    """

    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """
        Learn optimal temperature from validation data.

        Args:
            logits: Model output logits (N, C)
            labels: Ground truth labels (N,)
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        print(f"Learned temperature: {self.temperature.item():.4f}")

    def get_calibrated_confidence(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibrated confidence scores.

        Returns:
            (predictions, calibrated_confidences)
        """
        scaled_logits = self.forward(logits)
        probs = F.softmax(scaled_logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        return predictions, confidence


class PlattScaling(nn.Module):
    """
    Platt Scaling - Logistic regression on logits.

    More flexible than temperature scaling:
    calibrated = sigmoid(a * logit + b)

    Learns two parameters (a, b) per class.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling."""
        return torch.sigmoid(self.a * logits + self.b)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
    ):
        """Learn Platt parameters from validation data."""
        optimizer = torch.optim.LBFGS(
            [self.a, self.b],
            lr=0.01,
            max_iter=max_iter
        )

        # Convert labels to one-hot
        num_classes = logits.shape[-1]
        labels_onehot = F.one_hot(labels, num_classes).float()

        def closure():
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = F.binary_cross_entropy(probs, labels_onehot)
            loss.backward()
            return loss

        optimizer.step(closure)


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Key idea: Run inference multiple times with dropout enabled.
    The variance in predictions indicates model uncertainty.

    High variance = high epistemic uncertainty = model is unsure
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize MC Dropout.

        Args:
            model: Neural network with dropout layers
            n_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, UncertaintyBreakdown]:
        """
        Make prediction with uncertainty estimation.

        Args:
            x: Input tensor

        Returns:
            (mean_prediction, uncertainty_breakdown)
        """
        self.model.eval()
        self._enable_dropout()

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(x)
                predictions.append(F.softmax(output, dim=-1))

        predictions = torch.stack(predictions)  # (n_samples, batch, classes)

        # Mean prediction (expected output)
        mean_pred = predictions.mean(dim=0)

        # Epistemic uncertainty: variance across samples
        # High variance = model disagrees with itself
        epistemic = predictions.var(dim=0).mean(dim=-1)

        # Aleatoric uncertainty: entropy of mean prediction
        # High entropy = prediction is spread across classes
        aleatoric = -torch.sum(
            mean_pred * torch.log(mean_pred + 1e-10),
            dim=-1
        )

        # Total uncertainty
        total = epistemic + aleatoric

        # Determine if review needed
        needs_review = total.mean().item() > 0.5
        reason = ""
        if epistemic.mean() > aleatoric.mean():
            reason = "High model uncertainty - may need more training data"
        elif aleatoric.mean() > 0.7:
            reason = "High data uncertainty - ambiguous document"

        breakdown = UncertaintyBreakdown(
            epistemic=epistemic.mean().item(),
            aleatoric=aleatoric.mean().item(),
            total=total.mean().item(),
            needs_review=needs_review,
            reason=reason,
        )

        return mean_pred, breakdown


class EnsembleUncertainty:
    """
    Uncertainty estimation from model ensemble disagreement.

    If multiple models disagree, we're uncertain.
    If they agree, we're confident.
    """

    def __init__(self, models: List[nn.Module]):
        self.models = models

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Get prediction and uncertainty from ensemble.

        Returns:
            (mean_prediction, disagreement_score)
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                predictions.append(F.softmax(output, dim=-1))

        predictions = torch.stack(predictions)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Disagreement: variance across models
        disagreement = predictions.var(dim=0).mean().item()

        return mean_pred, disagreement


class CalibratedConfidenceScorer:
    """
    Complete calibrated confidence scoring system.

    Combines:
    1. Temperature scaling for base calibration
    2. MC Dropout for uncertainty estimation
    3. Confidence routing for human-in-the-loop

    This is what a Senior AI Engineer implements for production.
    """

    def __init__(
        self,
        base_model: nn.Module = None,
        temperature: float = 1.5,
        high_threshold: float = 0.9,
        low_threshold: float = 0.7,
        n_mc_samples: int = 10,
    ):
        """
        Initialize calibrated scorer.

        Args:
            base_model: Model to calibrate (optional)
            temperature: Initial temperature for scaling
            high_threshold: Auto-approve threshold
            low_threshold: Manual review threshold
            n_mc_samples: MC Dropout samples
        """
        self.base_model = base_model
        self.temp_scaler = TemperatureScaling(temperature)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.n_mc_samples = n_mc_samples

        if base_model:
            self.mc_dropout = MCDropoutUncertainty(base_model, n_mc_samples)
        else:
            self.mc_dropout = None

    def calibrate(
        self,
        validation_logits: torch.Tensor,
        validation_labels: torch.Tensor,
    ):
        """
        Calibrate confidence using validation data.

        This should be done before production deployment.
        """
        self.temp_scaler.calibrate(validation_logits, validation_labels)

    def score(
        self,
        logits: torch.Tensor = None,
        raw_confidence: float = None,
        field_completeness: float = None,
        validation_passed: bool = True,
    ) -> CalibrationResult:
        """
        Calculate calibrated confidence score.

        Can work with:
        1. Model logits (full calibration)
        2. Raw confidence score (simple scaling)
        3. Field-level metrics (heuristic)
        """
        # If we have logits, do full calibration
        if logits is not None:
            scaled_logits = self.temp_scaler(logits)
            probs = F.softmax(scaled_logits, dim=-1)
            calibrated_conf = probs.max().item()
            raw_conf = F.softmax(logits, dim=-1).max().item()

            # Get uncertainty from MC Dropout
            if self.mc_dropout is not None:
                _, uncertainty = self.mc_dropout.predict_with_uncertainty(logits)
                epistemic = uncertainty.epistemic
                aleatoric = uncertainty.aleatoric
            else:
                epistemic = 0.0
                aleatoric = 0.0

        # Otherwise use raw confidence with simple scaling
        elif raw_confidence is not None:
            raw_conf = raw_confidence
            # Apply temperature-like scaling
            temp = self.temp_scaler.temperature.item()
            calibrated_conf = raw_conf ** (1 / temp)
            calibrated_conf = min(calibrated_conf, 0.99)

            epistemic = 0.0
            aleatoric = 1.0 - raw_conf

        # Field completeness based scoring
        elif field_completeness is not None:
            raw_conf = field_completeness
            calibrated_conf = field_completeness * 0.9  # Conservative
            epistemic = 0.0
            aleatoric = 1.0 - field_completeness

        else:
            raise ValueError("Must provide logits, raw_confidence, or field_completeness")

        # Adjust for validation
        if not validation_passed:
            calibrated_conf *= 0.8

        total_uncertainty = epistemic + aleatoric

        return CalibrationResult(
            calibrated_confidence=calibrated_conf,
            raw_confidence=raw_conf,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_uncertainty,
        )

    def get_routing_decision(
        self,
        calibration_result: CalibrationResult,
    ) -> Dict:
        """
        Get routing decision based on calibrated confidence.

        Returns:
            Dict with status, action, and reasoning
        """
        conf = calibration_result.calibrated_confidence

        if conf >= self.high_threshold:
            return {
                'status': 'high',
                'action': 'auto_approve',
                'needs_review': False,
                'reason': f'High confidence ({conf:.2f})',
            }

        elif conf >= self.low_threshold:
            return {
                'status': 'medium',
                'action': 'sample_review',
                'needs_review': True,
                'priority': 'normal',
                'reason': f'Medium confidence ({conf:.2f}), recommend verification',
            }

        else:
            # Check uncertainty breakdown
            if calibration_result.epistemic_uncertainty > calibration_result.aleatoric_uncertainty:
                reason = f'Low confidence ({conf:.2f}), high model uncertainty'
            else:
                reason = f'Low confidence ({conf:.2f}), ambiguous document'

            return {
                'status': 'low',
                'action': 'manual_review',
                'needs_review': True,
                'priority': 'high',
                'reason': reason,
            }

    def calculate_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures how well confidence matches accuracy:
        ECE = sum(|accuracy(bin) - confidence(bin)|) * weight(bin)

        Perfect calibration: ECE = 0
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_accuracy = accuracies[mask].mean()
                bin_confidence = confidences[mask].mean()
                bin_weight = mask.sum() / len(confidences)

                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def get_reliability_diagram(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """
        Generate data for reliability diagram.

        A reliability diagram shows how well calibrated a model is.
        Perfect calibration = diagonal line.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if mask.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(accuracies[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)

        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'ece': self.calculate_ece(confidences, accuracies, n_bins),
        }
