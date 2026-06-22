"""
llm_client.py
-------------
Thin provider-agnostic wrapper so the rest of the app (summarizer, compare,
citation) doesn't need to know whether it's talking to OpenAI or Gemini.

Provider is chosen via the LLM_PROVIDER env var ("openai" or "gemini"),
which app.py sets based on which API key the user enters in the sidebar.

Both providers expose one function: complete_json(prompt) -> dict, and
complete_text(prompt) -> str. Internally each provider asks for JSON output
where supported (OpenAI's response_format, Gemini's response_mime_type) so
the calling code can rely on getting parseable JSON back in the common case.
"""

import json
import os


def get_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openai").lower()


def _default_model(provider: str, want_json: bool) -> str:
    if provider == "gemini":
        return "gemini-2.5-flash"
    return "gpt-4o-mini"


# ---------------- OpenAI backend ----------------

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _openai_complete(prompt: str, want_json: bool, model: str, temperature: float) -> str:
    client = _get_openai_client()
    kwargs = {}
    if want_json:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content


# ---------------- Gemini backend ----------------

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _gemini_complete(prompt: str, want_json: bool, model: str, temperature: float) -> str:
    from google.genai import types
    client = _get_gemini_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if want_json else "text/plain",
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


# ---------------- Public API ----------------

def complete_text(prompt: str, temperature: float = 0.3, model: str = None) -> str:
    """Free-form text completion (no JSON parsing)."""
    provider = get_provider()
    model = model or _default_model(provider, want_json=False)
    if provider == "gemini":
        return _gemini_complete(prompt, want_json=False, model=model, temperature=temperature).strip()
    return _openai_complete(prompt, want_json=False, model=model, temperature=temperature).strip()


def complete_json(prompt: str, temperature: float = 0.2, model: str = None) -> dict:
    """JSON completion. Returns a parsed dict. If parsing fails, returns
    {"_raw": <raw text>} so callers can decide how to degrade gracefully."""
    provider = get_provider()
    model = model or _default_model(provider, want_json=True)
    if provider == "gemini":
        raw = _gemini_complete(prompt, want_json=True, model=model, temperature=temperature)
    else:
        raw = _openai_complete(prompt, want_json=True, model=model, temperature=temperature)

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Strip markdown code fences if the model added them despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            try:
                return json.loads(cleaned.strip())
            except (json.JSONDecodeError, TypeError):
                pass
        return {"_raw": raw}


def has_api_key() -> bool:
    provider = get_provider()
    if provider == "gemini":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return bool(os.environ.get("OPENAI_API_KEY"))
