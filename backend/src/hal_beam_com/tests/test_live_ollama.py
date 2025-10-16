"""
Live-call test for local Ollama models.

This test prints the exact System Prompt used and makes a real call to the
local Ollama server (default http://localhost:11434). It is skipped automatically
if the Ollama server is not reachable or the model is not available.

Run with `-s` to see printed output.
"""

from __future__ import annotations

import os
import sys
import json
import re
import urllib.request
import urllib.error
import pytest

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

# Configure project path to import hal_beam_com.utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from hal_beam_com.utils import execute_llm_call  # noqa: E402


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = "gpt-oss:120b"


def _ollama_server_available() -> bool:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/version"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_model_present(model: str) -> bool:
    """Checks if the given model appears in the local Ollama tag list."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
            models = {m.get("name") for m in data.get("models", [])}
            return model in models
    except Exception:
        return False


def _extract_template(modelfile: str) -> str | None:
    """Extract a triple-quoted TEMPLATE block from an Ollama Modelfile."""
    try:
        m = re.search(
            r'(?is)^\s*(?:template|TEMPLATE)\s+("""|\'\'\'|`{3})\s*(.*?)\s*\1',
            modelfile,
            re.MULTILINE,
        )
        if m:
            return m.group(2).strip()
    except Exception:
        pass
    return None


def _ollama_show_template(model: str) -> tuple[str | None, str | None]:
    """
    Query Ollama /api/show for a model and try to extract its prompt template.

    Returns:
        (template, modelfile): either may be None if unavailable.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/show"
    try:
        payload = json.dumps({"name": model}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return None, None
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except Exception:
                # Some versions may return plain text; fall back
                return None, raw
        tmpl = None
        modelfile = None
        if isinstance(data, dict):
            modelfile = data.get("modelfile") or data.get("model") or data.get("details", {}).get("modelfile")
            tmpl = data.get("template") or data.get("details", {}).get("template")
            if not tmpl and isinstance(modelfile, str):
                tmpl = _extract_template(modelfile)
        return tmpl, modelfile
    except Exception:
        return None, None


@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_live_ollama_model(model_name: str):
    if not _ollama_server_available():
        pytest.skip("Ollama server not reachable at "
                    f"{OLLAMA_BASE_URL} – skipping live Ollama test")

    if not _ollama_model_present(model_name):
        pytest.skip(f"Ollama model '{model_name}' not found locally – "
                    "run `ollama pull {model_name}` and re-run the test")

    # Build a chat prompt template with a clear system instruction
    system_prompt = (
        "You are a minimal test runner for Ollama models. "
        "Reply with exactly the single word: test"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    # Print system prompt used
    messages = prompt_template.format_messages(text="Please reply with simply 'test'")
    sys_prompt_text = messages[0].content if messages else ""
    print("System prompt:\n" + sys_prompt_text)

    # Also show the full message list we are sending (role + first 200 chars)
    for idx, m in enumerate(messages):
        role = m.type if getattr(m, "type", "") != "human" else "user"
        try:
            content_preview = (m.content[:200] + "…") if isinstance(m.content, str) and len(m.content) > 200 else m.content
        except Exception:
            content_preview = str(m.content)
        print(f"Message {idx} → role={role}: {content_preview!r}")

    # Fetch and print model template/modelfile from Ollama for comparison
    tmpl, modelfile = _ollama_show_template(model_name)
    if tmpl:
        print("Ollama /api/show template:\n" + tmpl)
    elif modelfile:
        print("Ollama /api/show modelfile:\n" + modelfile)
    else:
        print("Could not retrieve template/modelfile from Ollama /api/show")

    # Create ChatOllama client directly (no env keys required)
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        temperature=0,
    )

    # Execute the call using the shared utility to keep behavior consistent
    output = execute_llm_call(
        llm,
        prompt_template,
        user_prompt="Please reply with simply 'test'",
        strip_markdown=True,
    )

    print("Output:", output)

    # Basic sanity checks, similar to other live tests
    assert isinstance(output, str) and output.strip(), "No output returned"
    assert any(tok in output.lower() for tok in ["test", "'test'"]), (
        f"Ollama model '{model_name}' did not produce expected token – "
        f"got: {output!r}"
    )
