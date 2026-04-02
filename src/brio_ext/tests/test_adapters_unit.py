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

    # Systems collapsed into first block, user turn follows, then open assistant turn
    assert prompt.startswith("<|im_start|>system")
    assert "<|im_start|>user\n" in prompt
    assert "<|im_start|>assistant\n" in prompt
    # No <out> fencing — fencing is handled at the factory level
    assert "<out>" not in prompt
    # Only ChatML stop token; </out> is NOT a stop token (fencing moved to factory)
    assert rendered["stop"] == ["<|im_end|>"]


def test_qwen_adapter_no_think_prefill():
    """When no_think=True, assistant turn is prefilled to skip reasoning."""
    adapter = QwenAdapter()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Rewrite the clause."},
    ]

    rendered = adapter.render(messages, no_think=True)
    prompt = rendered["prompt"]

    # Assistant turn prefilled with empty think block to suppress reasoning
    assert "<think>\n\n</think>" in prompt
    # The user content must be present unchanged (no sentinel stripping needed)
    assert "Rewrite the clause." in prompt
    # No sentinel in the prompt
    assert "/no_think" not in prompt


def test_qwen_adapter_no_think_default_disabled():
    """By default (no_think=False), assistant turn is NOT prefilled."""
    adapter = QwenAdapter()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Rewrite the clause."},
    ]

    rendered = adapter.render(messages)
    prompt = rendered["prompt"]

    # Normal assistant start — no prefill
    assert "<think>\n\n</think>" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_llama_adapter_passes_messages_for_native_template():
    """LlamaAdapter returns messages in OpenAI format for llamacpp native chat template."""
    adapter = LlamaAdapter()
    assert adapter.can_handle("llama-3.1-8b")

    rendered = adapter.render(MESSAGES)

    # Returns messages dict (not a raw prompt string) so llamacpp applies its template
    assert "messages" in rendered
    assert rendered["messages"] == MESSAGES
    # Llama 3.1+ stop tokens
    assert "<|eot_id|>" in rendered["stop"]
    assert "<|end_of_text|>" in rendered["stop"]
    # No <out> fencing — handled at factory level
    assert "prompt" not in rendered


def test_mistral_adapter_renders_inst_prompt():
    """MESSAGES has two system msgs — only messages[0] is extracted, second is silently dropped."""
    adapter = MistralAdapter()
    assert adapter.can_handle("mistral-7b-instruct")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("<s>")
    assert "[/INST]" in prompt
    # Only first system message is extracted; second one is silently dropped
    assert "System One" in prompt
    assert "System Two" not in prompt
    # No <out> fencing — handled at factory level
    assert "<out>" not in prompt
    assert "</s>" in rendered["stop"]


def test_mistral_adapter_single_user_message():
    adapter = MistralAdapter()
    rendered = adapter.render([{"role": "user", "content": "Hello"}])
    assert rendered["prompt"] == "<s>[INST] Hello[/INST]"


def test_mistral_adapter_multi_turn_with_assistant():
    """Format: <s>[INST] user[/INST] assistant</s>[INST] user2[/INST]"""
    adapter = MistralAdapter()
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    rendered = adapter.render(messages)
    assert rendered["prompt"] == "<s>[INST] What is 2+2?[/INST] 4</s>[INST] And 3+3?[/INST]"


def test_mistral_adapter_system_on_last_user_turn():
    """System message is prepended to the LAST user turn per the Jinja template."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    rendered = adapter.render(messages)
    expected = "<s>[INST] What is 2+2?[/INST] 4</s>[INST] Be concise.\n\nAnd 3+3?[/INST]"
    assert rendered["prompt"] == expected


def test_mistral_adapter_system_single_turn():
    """With only one user message, system is prepended to that (only) turn."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    rendered = adapter.render(messages)
    assert rendered["prompt"] == "<s>[INST] You are helpful.\n\nHello[/INST]"


def test_mistral_adapter_many_turns():
    adapter = MistralAdapter()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good."},
        {"role": "user", "content": "Great"},
    ]
    rendered = adapter.render(messages)
    expected = (
        "<s>[INST] Hi[/INST] Hello!</s>"
        "[INST] How are you?[/INST] Good.</s>"
        "[INST] Great[/INST]"
    )
    assert rendered["prompt"] == expected


def test_mistral_adapter_system_appears_once():
    """System text should only appear on the last user turn, exactly once."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Second"},
    ]
    rendered = adapter.render(messages)
    assert rendered["prompt"].count("System prompt.") == 1
    expected = "<s>[INST] First[/INST] Reply</s>[INST] System prompt.\n\nSecond[/INST]"
    assert rendered["prompt"] == expected


def test_gemma_adapter_renders_turns():
    adapter = GemmaAdapter()
    assert adapter.can_handle("gemma-2-9b")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    assert prompt.startswith("<start_of_turn>system\n")
    # Open model turn at the end (no <out> prefix — fencing at factory level)
    assert "<start_of_turn>model\n" in prompt
    assert "<out>" not in prompt
    # Gemma stop token
    assert "<end_of_turn>" in rendered["stop"]
    # No </out> stop token — handled at factory level
    assert "</out>" not in rendered["stop"]


def test_phi_adapter_renders_chatml_prompt():
    """PhiAdapter renders a ChatML prompt (not a passthrough messages dict)."""
    adapter = PhiAdapter()
    assert adapter.can_handle("phi-4-mini-instruct")

    rendered = adapter.render(MESSAGES)
    prompt = rendered["prompt"]

    # ChatML format with all message turns
    assert "<|im_start|>system\n" in prompt
    assert "<|im_start|>user\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    # No <out> fencing — handled at factory level
    assert "<out>" not in prompt


def test_phi_adapter_stop_tokens():
    """PhiAdapter includes both ChatML and legacy Phi closing tags; NOT <|end|>."""
    adapter = PhiAdapter()
    rendered = adapter.render(MESSAGES)

    # Phi-4 Mini may close with either token
    assert "<|im_end|>" in rendered["stop"]
    assert "</assistant>" in rendered["stop"]
    # <|end|> is Phi-4's native EOS — triggers immediately in ChatML context → blank response
    assert "<|end|>" not in rendered["stop"]
    # No <out>/</ out> fencing — handled at factory level
    assert "</out>" not in rendered["stop"]


# ---------------------------------------------------------------------------
# clean_response() — adapter post-processing
# ---------------------------------------------------------------------------

class TestLlamaAdapterCleanResponse:
    """LlamaAdapter.clean_response() strips Llama format markers."""

    def setup_method(self):
        self.adapter = LlamaAdapter()

    def test_strips_inst_markers(self):
        text = "[INST] What is a contract? [/INST] A contract is a legal agreement."
        result = self.adapter.clean_response(text)
        assert "[INST]" not in result
        assert "[/INST]" not in result
        assert "A contract is a legal agreement." in result

    def test_strips_sys_markers(self):
        text = "<<SYS>>You are helpful.<</SYS>>Here is my answer."
        result = self.adapter.clean_response(text)
        assert "<<SYS>>" not in result
        assert "<</SYS>>" not in result
        assert "Here is my answer." in result

    def test_strips_llama3_special_tokens(self):
        text = "<|start_header_id|>assistant<|end_header_id|>\n\nHere is the answer.<|eot_id|>"
        result = self.adapter.clean_response(text)
        assert "<|start_header_id|>" not in result
        assert "<|end_header_id|>" not in result
        assert "<|eot_id|>" not in result
        assert "Here is the answer." in result

    def test_preserves_newlines(self):
        """Newlines must survive clean_response so markdown formatting is intact."""
        text = "# Heading\n\nParagraph one.\n\n- Item 1\n- Item 2\n"
        result = self.adapter.clean_response(text)
        assert "\n" in result
        assert "# Heading" in result
        assert "- Item 1" in result

    def test_collapses_spaces_not_newlines(self):
        """Multiple spaces/tabs are collapsed; newlines are preserved."""
        text = "Word1   \t  Word2\nWord3"
        result = self.adapter.clean_response(text)
        assert "\n" in result
        assert "Word1 Word2" in result
        assert "Word3" in result

    def test_plain_text_passes_through(self):
        text = "This is a normal response."
        result = self.adapter.clean_response(text)
        assert result == text


class TestPhiAdapterCleanResponse:
    """PhiAdapter.clean_response() strips ChatML and Phi-specific markers."""

    def setup_method(self):
        self.adapter = PhiAdapter()

    def test_strips_chatml_im_end(self):
        text = "Here is my answer.<|im_end|>"
        result = self.adapter.clean_response(text)
        assert "<|im_end|>" not in result
        assert "Here is my answer." in result

    def test_strips_assistant_closing_tag(self):
        """Phi-4 Mini sometimes emits </assistant> as a closing artefact."""
        text = "The answer is 42.</assistant>"
        result = self.adapter.clean_response(text)
        assert "</assistant>" not in result
        assert "The answer is 42." in result

    def test_strips_im_start(self):
        text = "<|im_start|>This should not appear."
        result = self.adapter.clean_response(text)
        assert "<|im_start|>" not in result

    def test_strips_endoftext(self):
        text = "Done.<|endoftext|>"
        result = self.adapter.clean_response(text)
        assert "<|endoftext|>" not in result
        assert "Done." in result

    def test_plain_text_passes_through(self):
        text = "A straightforward response."
        assert self.adapter.clean_response(text) == text


class TestQwenAdapterCleanResponse:
    """QwenAdapter.clean_response() strips ChatML markers."""

    def setup_method(self):
        self.adapter = QwenAdapter()

    def test_strips_im_end(self):
        text = "Here is the answer.<|im_end|>"
        result = self.adapter.clean_response(text)
        assert "<|im_end|>" not in result
        assert "Here is the answer." in result

    def test_strips_im_start(self):
        text = "<|im_start|>assistant\nSome response"
        result = self.adapter.clean_response(text)
        assert "<|im_start|>" not in result

    def test_strips_endoftext(self):
        text = "Final answer.<|endoftext|>"
        result = self.adapter.clean_response(text)
        assert "<|endoftext|>" not in result
        assert "Final answer." in result

    def test_plain_text_passes_through(self):
        text = "Clean response with no markers."
        assert self.adapter.clean_response(text) == text
