from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "qwen-dora-db"
DEFAULT_CONFIG_PATH = ROOT / "config.json"
MODEL_METADATA_FILENAME = "model_metadata.json"
CLASSIFIER_HEADS_FILENAME = "classifier_heads.pt"

VIOLATION_TYPES = ("profanity", "insult", "threat", "hate")
ABUSE_SEVERITY_LEVELS = ("none", "mild", "moderate", "severe")


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config = load_json(path)
    if not isinstance(config, dict):
        raise ValueError("Config root must be a JSON object.")
    return config


def resolve_hidden_size(config) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    raise ValueError("Could not resolve hidden_size from model config.")


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_torch_dtype(dtype_name: str, device: torch.device):
    normalized = dtype_name.lower()
    if normalized == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")
    return mapping[normalized]


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token


def torch_load(path: str | Path, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class QwenModerationModel(nn.Module):
    def __init__(self, backbone, *, hidden_size: int, base_model_name: str):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.base_model_name = base_model_name
        self.violation_head = nn.Linear(hidden_size, len(VIOLATION_TYPES))
        self.severity_head = nn.Linear(hidden_size, len(ABUSE_SEVERITY_LEVELS))
        self.quote_head = nn.Linear(hidden_size, 1)

    def set_pad_token_id(self, pad_token_id: int) -> None:
        self.backbone.config.pad_token_id = pad_token_id
        text_config = getattr(self.backbone.config, "text_config", None)
        if text_config is not None:
            text_config.pad_token_id = pad_token_id

    def load_head_state_dict(self, state: dict) -> None:
        self.violation_head.load_state_dict(state["violation_head"])
        self.severity_head.load_state_dict(state["severity_head"])
        self.quote_head.load_state_dict(state["quote_head"])

    def _pool_last_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = attention_mask.long().sum(dim=1) - 1
        positions = positions.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, positions]

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise RuntimeError("Model did not return hidden states.")

        pooled = self._pool_last_token(hidden_states[-1], attention_mask)
        pooled = pooled.to(self.violation_head.weight.dtype)
        return SimpleNamespace(
            violation_logits=self.violation_head(pooled),
            severity_logits=self.severity_head(pooled),
            quote_logits=self.quote_head(pooled).squeeze(-1),
        )


class ModerationPredictor:
    def __init__(
        self,
        *,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        base_model_name: str | None = None,
        dtype: str = "auto",
        max_length: int | None = None,
        batch_size: int = 4,
        violation_threshold: float | None = None,
        quote_threshold: float | None = None,
    ):
        self.model_dir = Path(model_dir).resolve()
        self.metadata = load_json(self.model_dir / MODEL_METADATA_FILENAME)
        self.device = select_device()
        self.dtype = resolve_torch_dtype(dtype, self.device)
        self.max_length = int(max_length or self.metadata.get("max_length", 256))
        self.batch_size = int(batch_size)
        thresholds = self.metadata.get("thresholds", {})
        self.violation_threshold = float(
            violation_threshold
            if violation_threshold is not None
            else thresholds.get("violation_threshold", 0.5)
        )
        self.quote_threshold = float(
            quote_threshold
            if quote_threshold is not None
            else thresholds.get("quote_threshold", 0.5)
        )

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        ensure_tokenizer_padding(self.tokenizer)

        base_name = base_model_name or self.metadata["base_model_name"]
        backbone = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
        if (self.model_dir / "adapter_config.json").exists():
            backbone = PeftModel.from_pretrained(backbone, self.model_dir)

        self.model = QwenModerationModel(
            backbone,
            hidden_size=resolve_hidden_size(backbone.config),
            base_model_name=base_name,
        )
        head_state = torch_load(
            self.model_dir / CLASSIFIER_HEADS_FILENAME,
            map_location=self.device,
        )
        self.model.load_head_state_dict(head_state)
        self.model.set_pad_token_id(self.tokenizer.pad_token_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: Sequence[str]) -> list[dict]:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        results: list[dict] = []
        for offset in range(0, len(cleaned_texts), self.batch_size):
            batch_texts = cleaned_texts[offset : offset + self.batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.backends.cudnn.flags(enabled=False):
                outputs = self.model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                )
            violation_probs = torch.sigmoid(outputs.violation_logits).cpu()
            severity_probs = torch.softmax(outputs.severity_logits, dim=-1).cpu()
            quote_probs = torch.sigmoid(outputs.quote_logits).cpu()

            for index, text in enumerate(batch_texts):
                results.append(
                    build_prediction(
                        text=text,
                        violation_probabilities=violation_probs[index].tolist(),
                        severity_probabilities=severity_probs[index].tolist(),
                        quote_probability=quote_probs[index].item(),
                        violation_threshold=self.violation_threshold,
                        quote_threshold=self.quote_threshold,
                    )
                )
        return results


def build_prediction(
    *,
    text: str,
    violation_probabilities: Sequence[float],
    severity_probabilities: Sequence[float],
    quote_probability: float,
    violation_threshold: float,
    quote_threshold: float,
) -> dict:
    violation_scores = {
        label: round(float(probability), 4)
        for label, probability in zip(VIOLATION_TYPES, violation_probabilities)
    }
    selected_violations = [
        label
        for label, probability in violation_scores.items()
        if probability >= violation_threshold
    ]

    severity_scores = {
        label: round(float(probability), 4)
        for label, probability in zip(ABUSE_SEVERITY_LEVELS, severity_probabilities)
    }
    predicted_severity = max(severity_scores, key=severity_scores.get)
    if not selected_violations:
        predicted_severity = "none"

    quote_probability = round(float(quote_probability), 4)
    return {
        "text": text,
        "violation_types": selected_violations,
        "violation_type_scores": violation_scores,
        "abuse_severity": predicted_severity,
        "abuse_severity_scores": severity_scores,
        "is_quoted_or_explanatory": quote_probability >= quote_threshold,
        "is_quoted_or_explanatory_score": quote_probability,
        "is_clean": not selected_violations,
    }


def load_texts_from_file(path: str | Path) -> list[str]:
    texts = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{path}:{line_number}: missing non-empty text field.")
                texts.append(text)
            else:
                texts.append(line)
    return texts


def build_predictor_from_config(config: dict) -> ModerationPredictor:
    model_config = config.get("model", {})
    runtime_config = config.get("runtime", {})
    threshold_config = config.get("thresholds", {})
    if not isinstance(model_config, dict):
        raise ValueError("config.model must be an object.")
    if not isinstance(runtime_config, dict):
        raise ValueError("config.runtime must be an object.")
    if not isinstance(threshold_config, dict):
        raise ValueError("config.thresholds must be an object.")

    model_dir = resolve_config_path(
        ROOT,
        model_config.get("model_dir", "qwen-dora-db"),
    )
    return ModerationPredictor(
        model_dir=model_dir or DEFAULT_MODEL_DIR,
        base_model_name=model_config.get("base_model_name"),
        dtype=runtime_config.get("dtype", "auto"),
        max_length=runtime_config.get("max_length"),
        batch_size=runtime_config.get("batch_size", 4),
        violation_threshold=threshold_config.get("violation_threshold"),
        quote_threshold=threshold_config.get("quote_threshold"),
    )


_api_predictor: ModerationPredictor | None = None


def get_api_predictor() -> ModerationPredictor:
    global _api_predictor
    if _api_predictor is None:
        _api_predictor = build_predictor_from_config(load_config())
    return _api_predictor


try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field

    class PredictRequest(BaseModel):
        texts: list[str] = Field(min_length=1)

    app = FastAPI(title="NoMoreSwearing MVP")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    def predict_endpoint(request: PredictRequest) -> dict:
        return {"predictions": get_api_predictor().predict(request.texts)}

except ImportError:
    app = None


def main() -> int:
    if app is None:
        raise SystemExit("Install fastapi and uvicorn to run the API service.")
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
