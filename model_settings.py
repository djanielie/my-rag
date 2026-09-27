"""Discover installed Ollama chat models and persist the user's selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import CHAT_MODEL, DATA_DIR


SETTINGS_PATH = DATA_DIR / "settings.json"


def load_chat_model(path: Path = SETTINGS_PATH) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data.get("chat_model")
        return model.strip() if isinstance(model, str) and model.strip() else CHAT_MODEL
    except (OSError, ValueError, TypeError):
        return CHAT_MODEL


def save_chat_model(model: str, path: Path = SETTINGS_PATH) -> None:
    model = model.strip()
    if not model:
        raise ValueError("Chat model cannot be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"chat_model": model}, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    if hasattr(response, key):
        return getattr(response, key)
    if isinstance(response, dict):
        return response.get(key, default)
    return default


def installed_chat_models(client: Any) -> list[str]:
    """Return installed models that advertise text-completion support."""
    response = client.list()
    models = _response_value(response, "models", []) or []
    chat_models: list[str] = []

    for model_info in models:
        name = _response_value(model_info, "model") or _response_value(
            model_info, "name"
        )
        if not name:
            continue

        try:
            details = client.show(str(name))
            capabilities = _response_value(details, "capabilities", []) or []
            supports_chat = not capabilities or "completion" in capabilities
        except Exception:
            supports_chat = "embed" not in str(name).casefold()

        if supports_chat:
            chat_models.append(str(name))

    return sorted(set(chat_models), key=str.casefold)
