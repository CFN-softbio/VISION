"""
Live-call tests for external LLM providers.
Currently covers Abacus-AI; more providers can be appended
to `PROVIDERS` later.

The test is skipped automatically when the relevant
API-key env-var is missing, so it is safe in CI.
"""

from __future__ import annotations

import os
import pytest
import sys

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage

# Configure project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from hal_beam_com.utils import load_model, execute_llm_call


# --------------------------------------------------------------------------- #
#  Configuration of live providers to test
# --------------------------------------------------------------------------- #
PROVIDERS = [
    # {
    #     "name": "abacus",
    #     "model": "gpt-4o-abacus",
    #     "env_var": "ABACUS_API_KEY",
    #     "user_prompt": "Please reply with simply 'test'",
    #     "expect": ["test", "'test'"],
    # },
    {
        "name": "openrouter_grok_4",
        "model": "grok-4-or",
        "env_var": "OPENROUTER_API_KEY",
        "user_prompt": "Please reply with simply 'test'",
        "expect": ["test", "'test'"],
    },
    {
        "name": "openrouter_devstral_medium",
        "model": "devstral-med-or",
        "env_var": "OPENROUTER_API_KEY",
        "user_prompt": "Please reply with simply 'test'",
        "expect": ["test", "'test'"],
    },
    # {
    #     "name"      : "ollama_qwen3",
    #     "model"     : "qwen3",
    #     # no API-key needed; leave env blank so the test always runs
    #     "user_prompt": "How would you derive Pythagoras, come up with 3 distinct ways.",
    #     "expect"    : None,
    # },
    {
        "name": "openrouter_grok_code_fast_1",
        "model": "grok-code-fast-1-or",
        "env_var": "OPENROUTER_API_KEY",
        "user_prompt": "Please reply with simply 'test'",
        "expect": ["test", "'test'"],
    },
    {
        "name": "openrouter_qwen3_coder_480b",
        "model": "qwen3-coder-480b-or",
        "env_var": "OPENROUTER_API_KEY",
        "user_prompt": "Please reply with simply 'test'",
        "expect": ["test", "'test'"],
    },
    {
        "name": "openrouter_sonoma_sky_alpha",
        "model": "sonoma-sky-alpha-or",
        "env_var": "OPENROUTER_API_KEY",
        "user_prompt": "Please reply with simply 'test'",
        "expect": ["test", "'test'"],
    },
    # {
    #     "name"      : "ollama_qwen3_thinking",
    #     "model"     : "qwen3-thinking",
    #     "user_prompt": "How would you derive Pythagoras, come up with 3 distinct ways.",
    #     "expect"    : None,
    # },
    # {
    #     "name"      : "azure_o3_high",
    #     "model"     : "o3-high",
    #     "user_prompt": "How would you derive Pythagoras, come up with 3 distinct ways.",
    #     "expect"    : None,
    # },
    # {
    #     "name"       : "abacus_claude_v4_sonnet",
    #     "model"      : "claude-v4-sonnet-abacus",
    #     "env_var"    : "ABACUS_API_KEY",
    #     "user_prompt": "Please reply with simply 'test'",
    #     "expect"     : ["test", "'test'"],
    # },
    # {
    #     "name"       : "abacus_claude_v4_opus",
    #     "model"      : "claude-v4-opus-abacus",
    #     "env_var"    : "ABACUS_API_KEY",
    #     "user_prompt": "Please reply with simply 'test'",
    #     "expect"     : ["test", "'test'"],
    # },
    # {
    #     "name"       : "abacus_gemini_2_5_pro",
    #     "model"      : "gemini-2-5-pro-abacus",
    #     "env_var"    : "ABACUS_API_KEY",
    #     "user_prompt": "Please reply with simply 'test'",
    #     "expect"     : ["test", "'test'"],
    # },
    # Add future providers here …
]


# --------------------------------------------------------------------------- #
#  Parameterised live-test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", PROVIDERS, ids=[p["name"] for p in PROVIDERS])
def test_live_llm_provider(cfg: dict):
    """
    Make a real request to the provider and verify we get *some*
    sensible answer back.
    """

    required = cfg.get("env_vars") or [cfg.get("env_var")]
    missing  = [v for v in required if v and not os.getenv(v)]
    if missing:
        pytest.skip(f"{', '.join(missing)} not set – skipping live call")

    # 1) load model
    llm = load_model(cfg["model"])

    # 2) minimal prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="Answer concisely."),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    # 3) run the call
    output = execute_llm_call(
        llm,
        prompt_template,
        user_prompt=cfg["user_prompt"],
        strip_markdown=True,
    )

    print("Output:", output)

    # 4) sanity checks
    assert isinstance(output, str) and output.strip(), "No output returned"

    if cfg["expect"] is not None:
        assert any(word.lower() in output.lower() for word in cfg["expect"]), (
        f"Provider '{cfg['name']}' did not mention an expected token "
        f"in its answer: {output!r}"
        )
