from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    confidence: float
    provider: str


class PretrainedIntentModel:
    """Optional Hugging Face inference wrapper with deterministic fallback."""

    def __init__(self) -> None:
        self.provider = "rules"
        self.classifier = self._load_transformers_pipeline()

    def predict(self, text: str) -> IntentPrediction:
        if self.classifier:
            labels = ["sensor_selection", "power_design", "microcontroller_selection", "general_component_search"]
            result = self.classifier(text, candidate_labels=labels)
            return IntentPrediction(
                label=str(result["labels"][0]),
                confidence=float(result["scores"][0]),
                provider=self.provider,
            )

        lowered = text.lower()
        if any(term in lowered for term in ["sensor", "temperatura", "humedad", "presion"]):
            return IntentPrediction("sensor_selection", 0.72, self.provider)
        if any(term in lowered for term in ["regulador", "fuente", "3.3", "alimentar", "power"]):
            return IntentPrediction("power_design", 0.68, self.provider)
        if any(term in lowered for term in ["wifi", "bluetooth", "iot", "esp32", "microcontrolador"]):
            return IntentPrediction("microcontroller_selection", 0.74, self.provider)
        return IntentPrediction("general_component_search", 0.55, self.provider)

    def _load_transformers_pipeline(self):
        try:
            from transformers import pipeline

            self.provider = "facebook/bart-large-mnli"
            return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception:
            return None
