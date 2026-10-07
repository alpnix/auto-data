"""
LLM access shared by the setup agent and the research pipeline.

Two provider families are supported behind one tiny interface:
  openai   any OpenAI-compatible HTTP API (OpenAI, DeepSeek), authenticated with an API key
  bedrock  Amazon Bedrock via the Converse API, authenticated with normal AWS credentials
           (AWS_PROFILE / access keys / instance role) — or, if AWS_BEARER_TOKEN_BEDROCK is
           set, via Bedrock's OpenAI-compatible endpoint with that key

Components:
  MODEL_CONFIGS         registry of supported chat models and how to reach them
  get_available_models  models that can be used right now (key or AWS credentials present)
  build_llm_client      returns an object with .complete(system, user, max_tokens) -> str
  chat_text / chat_json one call returning text, or parsed JSON (None if not valid JSON)
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

# provider "openai": base_url + env_keys (first non-empty env var is the API key).
# provider "bedrock": the key is the Bedrock model / inference-profile id.
# use_completion_tokens: newer OpenAI models reject max_tokens in favour of max_completion_tokens.
MODEL_CONFIGS: Dict[str, Dict] = {
    # ── Amazon Bedrock (cheapest first) ──
    "us.amazon.nova-micro-v1:0":                      {"provider": "bedrock", "label": "Bedrock · Nova Micro"},
    "us.amazon.nova-lite-v1:0":                       {"provider": "bedrock", "label": "Bedrock · Nova Lite"},
    "us.amazon.nova-pro-v1:0":                        {"provider": "bedrock", "label": "Bedrock · Nova Pro"},
    "us.anthropic.claude-haiku-4-5-20251001-v1:0":    {"provider": "bedrock", "label": "Bedrock · Claude Haiku 4.5"},
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0":   {"provider": "bedrock", "label": "Bedrock · Claude Sonnet 4.5"},
    "deepseek.v3.2":                                  {"provider": "bedrock", "label": "Bedrock · DeepSeek V3.2"},
    "us.meta.llama3-3-70b-instruct-v1:0":             {"provider": "bedrock", "label": "Bedrock · Llama 3.3 70B"},
    # ── Direct APIs ──
    "deepseek-chat": {"provider": "openai", "label": "DeepSeek Chat (direct)", "base_url": "https://api.deepseek.com",
                      "env_keys": ["DEEPSEEK_API_KEY", "DEEPSEEK_API"]},
    "gpt-4o-mini":   {"provider": "openai", "label": "OpenAI GPT-4o Mini", "base_url": None, "env_keys": ["OPENAI_API_KEY"]},
    "gpt-5-nano":    {"provider": "openai", "label": "OpenAI GPT-5 Nano", "base_url": None, "env_keys": ["OPENAI_API_KEY"],
                      "use_completion_tokens": True},
}

DEFAULT_MODEL = "us.amazon.nova-lite-v1:0"


# ── credential discovery ─────────────────────────────────────────────────────

def _api_key_for(cfg: Dict) -> Optional[str]:
    return next((os.getenv(k) for k in cfg.get("env_keys", []) if os.getenv(k)), None)


def _aws_credentials_present() -> bool:
    """True if boto3 can resolve credentials (env vars, AWS_PROFILE, ~/.aws, instance role)."""
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        return True
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


def _is_available(cfg: Dict) -> bool:
    return _aws_credentials_present() if cfg["provider"] == "bedrock" else bool(_api_key_for(cfg))


def get_available_models() -> List[Dict]:
    """Models that can actually be used right now, in registry order."""
    return [{"id": mid, "label": cfg["label"], "provider": cfg["provider"]}
            for mid, cfg in MODEL_CONFIGS.items() if _is_available(cfg)]


# ── clients ──────────────────────────────────────────────────────────────────

class OpenAICompatClient:
    """OpenAI SDK against any compatible endpoint (OpenAI, DeepSeek, Bedrock's /openai/v1)."""

    def __init__(self, model_id: str, api_key: str, base_url: Optional[str], use_completion_tokens: bool = False):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.token_param = "max_completion_tokens" if use_completion_tokens else "max_tokens"

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **{self.token_param: max_tokens},
        )
        return response.choices[0].message.content or ""


class BedrockClient:
    """Amazon Bedrock Converse API with SigV4 (standard AWS credential chain)."""

    def __init__(self, model_id: str):
        import boto3
        self.client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        self.model_id = model_id

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        return "".join(part.get("text", "") for part in response["output"]["message"]["content"])


def build_llm_client(model_id: str):
    """Client for `model_id`; raises ValueError if the model is unknown or has no credentials."""
    cfg = MODEL_CONFIGS.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")
    if cfg["provider"] == "bedrock":
        bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bearer:
            return OpenAICompatClient(model_id, bearer, f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1")
        if not _aws_credentials_present():
            raise ValueError(f"No AWS credentials for {model_id} (set AWS_PROFILE or AWS_BEARER_TOKEN_BEDROCK)")
        return BedrockClient(model_id)
    api_key = _api_key_for(cfg)
    if not api_key:
        raise ValueError(f"No API key for {model_id} (tried {cfg['env_keys']})")
    return OpenAICompatClient(model_id, api_key, cfg.get("base_url"), cfg.get("use_completion_tokens", False))


# ── calls ────────────────────────────────────────────────────────────────────

def chat_text(client, model_id: str, system: str, user: str, max_tokens: int) -> str:
    """Single completion; returns the text with any ``` fences stripped."""
    text = client.complete(system, user, max_tokens)
    if "```" in text:
        text = text.split("```", 2)[1]
        text = text.split("\n", 1)[1] if text.startswith(("json", "python")) else text
    return text.strip()


def chat_json(client, model_id: str, system: str, user: str, max_tokens: int) -> Optional[Dict]:
    """Like chat_text but parsed as JSON; None if the reply is not valid JSON. API errors propagate."""
    try:
        return json.loads(chat_text(client, model_id, system, user, max_tokens))
    except json.JSONDecodeError:
        return None
