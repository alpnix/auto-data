"""
LLM access shared by the setup agent and the research pipeline.

Components:
  MODEL_CONFIGS         registry of supported chat models and how to reach them
  get_available_models  models whose API key is present in the environment
  build_llm_client      OpenAI-compatible client for a model id
  chat_json             one chat call that returns parsed JSON (or None)
  chat_text             one chat call that returns the raw text (or "")
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Each entry: where the API lives, which env vars may hold the key, and whether
# the model wants `max_completion_tokens` (newer OpenAI models) or `max_tokens`.
MODEL_CONFIGS: Dict[str, Dict] = {
    "deepseek-chat": {
        "label": "DeepSeek Chat (V3)",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "env_keys": ["DEEPSEEK_API_KEY", "DEEPSEEK_API"],
        "use_completion_tokens": False,
    },
    "gpt-5-nano": {
        "label": "GPT-5 Nano",
        "provider": "openai",
        "base_url": None,
        "env_keys": ["OPENAI_API_KEY"],
        "use_completion_tokens": True,
    },
    "gpt-4o-mini": {
        "label": "GPT-4o Mini",
        "provider": "openai",
        "base_url": None,
        "env_keys": ["OPENAI_API_KEY"],
        "use_completion_tokens": False,
    },
}

DEFAULT_MODEL = "deepseek-chat"


def _api_key_for(cfg: Dict) -> Optional[str]:
    """First non-empty env var among the model's accepted key names."""
    return next((os.getenv(k) for k in cfg["env_keys"] if os.getenv(k)), None)


def get_available_models() -> List[Dict]:
    """Models that can actually be used right now (key configured)."""
    return [
        {"id": mid, "label": cfg["label"], "provider": cfg["provider"]}
        for mid, cfg in MODEL_CONFIGS.items()
        if _api_key_for(cfg)
    ]


def build_llm_client(model_id: str) -> OpenAI:
    """Return an OpenAI-compatible client for `model_id`; raises ValueError if unusable."""
    cfg = MODEL_CONFIGS.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")
    api_key = _api_key_for(cfg)
    if not api_key:
        raise ValueError(f"No API key for {model_id} (tried {cfg['env_keys']})")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def _token_kwargs(model_id: str, n_tokens: int) -> Dict:
    """Pick the token-limit parameter name the model accepts."""
    if MODEL_CONFIGS.get(model_id, {}).get("use_completion_tokens"):
        return {"max_completion_tokens": n_tokens}
    return {"max_tokens": n_tokens}


def chat_text(client: OpenAI, model_id: str, system: str, user: str, max_tokens: int) -> str:
    """Single chat completion; returns the message text with any ``` fences stripped."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **_token_kwargs(model_id, max_tokens),
    )
    text = response.choices[0].message.content or ""
    if "```" in text:
        # Take the first fenced block, dropping an optional language tag.
        text = text.split("```", 2)[1]
        text = text.split("\n", 1)[1] if text.startswith(("json", "python")) else text
    return text.strip()


def chat_json(client: OpenAI, model_id: str, system: str, user: str, max_tokens: int) -> Optional[Dict]:
    """Like chat_text but parses the reply as JSON; returns None if it isn't valid JSON.
    API errors are not swallowed so callers can log them."""
    try:
        return json.loads(chat_text(client, model_id, system, user, max_tokens))
    except json.JSONDecodeError:
        return None
