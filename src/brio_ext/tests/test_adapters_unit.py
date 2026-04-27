"""Unit tests for brio_ext adapters."""

from pathlib import Path

import jinja2
import pytest

from brio_ext.adapters.gemma_adapter import Gemma4Adapter
from brio_ext.adapters.llama_adapter import LlamaAdapter
from brio_ext.adapters.mistral_adapter import MistralAdapter
from brio_ext.adapters.phi_adapter import PhiAdapter
from brio_ext.adapters.qwen_adapter import QwenAdapter

GEMMA_TEMPLATE_PATH = Path(__file__).parent / "resources" / "gemma_chat_template.jinja"
GEMMA_BOS = "<bos>"

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


def test_mistral_adapter_system_injected_as_first_turn():
    """System message is injected as a synthetic first user/assistant exchange."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    rendered = adapter.render(messages)
    expected = (
        "<s>[INST] Be concise.[/INST] OK, I will follow these instructions.</s>"
        "[INST] What is 2+2?[/INST] 4</s>"
        "[INST] And 3+3?[/INST]"
    )
    assert rendered["prompt"] == expected


def test_mistral_adapter_system_single_turn():
    """With only one user message, system is injected before it."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    rendered = adapter.render(messages)
    expected = (
        "<s>[INST] You are helpful.[/INST] OK, I will follow these instructions.</s>"
        "[INST] Hello[/INST]"
    )
    assert rendered["prompt"] == expected


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
    """System text should only appear once, in the synthetic first exchange."""
    adapter = MistralAdapter()
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Second"},
    ]
    rendered = adapter.render(messages)
    assert rendered["prompt"].count("System prompt.") == 1
    expected = (
        "<s>[INST] System prompt.[/INST] OK, I will follow these instructions.</s>"
        "[INST] First[/INST] Reply</s>"
        "[INST] Second[/INST]"
    )
    assert rendered["prompt"] == expected


class TestGemma4Adapter:
    """Gemma 4 adapter — rendered output is verified against the actual chat
    template shipped with `google/gemma-4-*` models. The template is checked
    into `resources/gemma_chat_template.jinja`; the adapter must produce the
    same string the template does (modulo the leading BOS token, which the
    engine prepends during tokenization).
    """

    @pytest.fixture(scope="class")
    def template(self):
        env = jinja2.Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        return env.from_string(GEMMA_TEMPLATE_PATH.read_text())

    @staticmethod
    def _render_template(template, messages, *, enable_thinking=False):
        rendered = template.render(
            bos_token=GEMMA_BOS,
            messages=messages,
            tools=None,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        # Adapter does not emit BOS — the engine adds it during tokenization.
        assert rendered.startswith(GEMMA_BOS)
        return rendered[len(GEMMA_BOS):]

    def test_can_handle(self):
        adapter = Gemma4Adapter()
        assert adapter.can_handle("gemma-4-26b-a4b-it")
        assert adapter.can_handle("Gemma-4-IT")
        assert adapter.can_handle("google/gemma4-9b")
        assert not adapter.can_handle("gemma-2-9b")
        assert not adapter.can_handle("gemma-3-27b")
        assert not adapter.can_handle("qwen2.5-7b-instruct")
        assert not adapter.can_handle("")
        assert not adapter.can_handle(None)

    def test_stop_sequence(self):
        rendered = Gemma4Adapter().render([{"role": "user", "content": "Hi"}], no_think=True)
        assert rendered["stop"] == ["<turn|>"]

    def test_single_user_no_thinking(self, template):
        messages = [{"role": "user", "content": "Hello"}]
        expected = self._render_template(template, messages, enable_thinking=False)
        rendered = Gemma4Adapter().render(messages, no_think=True)
        assert rendered["prompt"] == expected

    def test_system_and_user_no_thinking(self, template):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        expected = self._render_template(template, messages, enable_thinking=False)
        rendered = Gemma4Adapter().render(messages, no_think=True)
        assert rendered["prompt"] == expected

    def test_multi_turn_no_thinking(self, template):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me a joke."},
        ]
        expected = self._render_template(template, messages, enable_thinking=False)
        rendered = Gemma4Adapter().render(messages, no_think=True)
        assert rendered["prompt"] == expected

    def test_user_only_thinking_enabled(self, template):
        messages = [{"role": "user", "content": "Solve this problem"}]
        expected = self._render_template(template, messages, enable_thinking=True)
        rendered = Gemma4Adapter().render(messages, no_think=False)
        assert rendered["prompt"] == expected

    def test_system_user_thinking_enabled(self, template):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        expected = self._render_template(template, messages, enable_thinking=True)
        rendered = Gemma4Adapter().render(messages, no_think=False)
        assert rendered["prompt"] == expected

    def test_strips_thinking_from_assistant_history(self, template):
        """Assistant messages with leaked <|channel>...<channel|> blocks must be
        sanitized identically to the template's `strip_thinking` macro."""
        messages = [
            {"role": "user", "content": "Q1"},
            {
                "role": "assistant",
                "content": "<|channel>thought\nreasoning here<channel|>The answer is 42.",
            },
            {"role": "user", "content": "Q2"},
        ]
        expected = self._render_template(template, messages, enable_thinking=False)
        rendered = Gemma4Adapter().render(messages, no_think=True)
        assert rendered["prompt"] == expected

    def test_trims_message_whitespace(self, template):
        """Whitespace-padded content must be trimmed — matches the template's `| trim`."""
        messages = [
            {"role": "system", "content": "  Be concise.  \n"},
            {"role": "user", "content": "  Hi  "},
        ]
        expected = self._render_template(template, messages, enable_thinking=False)
        rendered = Gemma4Adapter().render(messages, no_think=True)
        assert rendered["prompt"] == expected


class TestGemma4AdapterCleanResponse:
    """Gemma4Adapter.clean_response() strips Gemma 4 turn and channel markers."""

    def setup_method(self):
        self.adapter = Gemma4Adapter()

    def test_strips_turn_markers(self):
        text = "<|turn>model\nHello<turn|>"
        assert self.adapter.clean_response(text) == "Hello"

    def test_strips_complete_thinking_block(self):
        text = "<|channel>thought\nreasoning here<channel|>The answer."
        assert self.adapter.clean_response(text) == "The answer."

    def test_drops_unclosed_thinking_channel(self):
        text = "answer<|channel>incomplete"
        assert self.adapter.clean_response(text) == "answer"

    def test_strips_think_marker(self):
        assert self.adapter.clean_response("<|think|>\nHello") == "Hello"

    def test_passes_through_clean_text(self):
        assert self.adapter.clean_response("Just a regular response.") == "Just a regular response."


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
