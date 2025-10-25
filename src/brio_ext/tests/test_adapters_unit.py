"""Unit tests for brio_ext adapters."""

from brio_ext.adapters.gemma_adapter import GemmaAdapter
from brio_ext.adapters.llama_adapter import LlamaAdapter
from brio_ext.adapters.mistral_adapter import MistralAdapter
from brio_ext.adapters.phi_adapter import PhiAdapter
from brio_ext.adapters.qwen_adapter import QwenAdapter

MESSAGES = [
    {"role": "system", "content": "System One"},
    {"role": "system", "content": "System Two"},
    {"role": "user", "content": "Rewrite the clause."},
]


def test_qwen_adapter_renders_chatml_prompt():
    adapter = QwenAdapter()
    assert adapter.can_handle("Qwen2.5-7B-Instruct")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("<|im_start|>system")
    assert "<|im_start|>assistant" in prompt
    assert prompt.endswith("<out>\n")
    assert "</out>" in rendered["stop"]
    assert "<|im_end|>" in rendered["stop"]


def test_llama_adapter_renders_inst_prompt():
    adapter = LlamaAdapter()
    assert adapter.can_handle("llama-3.1-8b")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("[INST] <<SYS>>")
    assert "<</SYS>>" in prompt
    assert prompt.rstrip().endswith("<out>")
    assert rendered["stop"] == ["</out>"]


def test_mistral_adapter_renders_inst_prompt():
    adapter = MistralAdapter()
    assert adapter.can_handle("mistral-7b-instruct")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("[INST] ")
    assert prompt.rstrip().endswith("<out>")
    assert rendered["stop"] == ["</out>"]


def test_gemma_adapter_renders_turns():
    adapter = GemmaAdapter()
    assert adapter.can_handle("gemma-2-9b")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("<start_of_turn>system\n")
    assert "<start_of_turn>model\n<out>" in prompt
    assert "</out>" in rendered["stop"]
    assert "<end_of_turn>" in rendered["stop"]


def test_phi_adapter_passes_through_messages():
    adapter = PhiAdapter()
    assert adapter.can_handle("phi-4-mini-instruct")

    rendered = adapter.render(MESSAGES)
    assert rendered["messages"] == MESSAGES
    assert rendered["stop"] == ["</out>"]
