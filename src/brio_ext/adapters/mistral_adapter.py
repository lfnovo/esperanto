"""Mistral family prompt adapter.

Implements the chat template for mistralai/Mistral-7B-Instruct-v0.3 as published
in the updated tokenizer_config.json on Hugging Face:
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/commit/43ee8f4afb6fc9e4304a8ed87aaa3a36a0e06939

NOTE: The GGUF we use (bartowski/Mistral-7B-Instruct-v0.3-GGUF, May 2024)
contains an older template that does NOT support system messages and has
different whitespace rules. The original mistralai repo was updated after the
GGUF was published. We follow the updated template from mistralai.

The Jinja template (formatted for readability):

    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content'] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set loop_messages = messages %}
    {%- endif %}

    {{- bos_token }}
    {%- for message in loop_messages %}
        {%- if message['role'] == 'user' %}
            {%- if loop.last and system_message is defined %}
                {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}
            {%- else %}
                {{- '[INST] ' + message['content'] + '[/INST]' }}
            {%- endif %}
        {%- elif message['role'] == 'assistant' %}
            {{- ' ' + message['content'] + eos_token }}
        {%- endif %}
    {%- endfor %}

Which renders as (bos=<s>, eos=</s>):

    <s>[INST] {user}[/INST] {assistant}</s>[INST] {system}\\n\\n{user2}[/INST]

Key details:
    - Space after [INST], NO space before [/INST]
    - Space BEFORE assistant content
    - System message is prepended to the LAST user turn (not the first)
    - <s> prefix (bos_token), </s> after each assistant turn (eos_token)

WARNING: This template is specific to mistralai/Mistral-7B-Instruct-v0.3.
Newer Mistral models (v7+) use different templates with different formatting.
Verify the GGUF chat template matches before using this adapter with other models:

    v7   (128k, Tekken)   Mistral Large, Pixtral, Mistral Small
    v11  (128k, Tekken)   Newer function-calling models
    v13  (128k, Tekken)   Models with think blocks (Mistral Medium, etc.)
    v15  (128k, Tekken)   Models with model-settings support

See: https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-chat_templates
"""

from __future__ import annotations

from typing import Dict, List

from brio_ext.adapters import ChatAdapter, RenderedPrompt


class MistralAdapter(ChatAdapter):
    """Render Esperanto messages using the Mistral 7B Instruct v0.3 template."""

    def can_handle(self, model_id: str) -> bool:
        return "mistral" in (model_id or "").lower()

    def render(self, messages: List[Dict[str, str]], no_think: bool = False) -> RenderedPrompt:
        # Extract optional leading system message (template only supports it
        # as the very first message)
        system_text = ""
        loop_messages = messages
        if messages and messages[0]["role"] == "system":
            system_text = messages[0]["content"]
            loop_messages = messages[1:]

        # Find the index of the last user message (system gets prepended there)
        last_user_idx = None
        for i, msg in enumerate(loop_messages):
            if msg["role"] == "user":
                last_user_idx = i

        parts: list[str] = []
        for i, msg in enumerate(loop_messages):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if i == last_user_idx and system_text:
                    parts.append(f"[INST] {system_text}\n\n{content}[/INST]")
                else:
                    parts.append(f"[INST] {content}[/INST]")
            elif role == "assistant":
                parts.append(f" {content}</s>")

        prompt = "<s>" + "".join(parts)
        return {"prompt": prompt, "stop": ["</s>"]}

    def clean_response(self, text: str) -> str:
        """Remove Mistral format markers from response."""
        cleaned = text
        for marker in ["[INST]", "[/INST]", "</s>", "<s>"]:
            cleaned = cleaned.replace(marker, "")
        return cleaned.strip()
