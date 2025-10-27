#!/usr/bin/env python3
"""
Manual test runner for brio_ext validation with full pipeline visibility.

Usage:
    python scripts/test_with_llm.py [scenario] [model]

Scenarios:
    pirate      - Simple system message test (pirate personality)
    inventor    - Medium context inventor lookup (tests Qwen system message bug)
    multiturn   - Multi-turn conversation
    all         - Run all scenarios

Models (numbered for convenience):
    1   - Qwen 2.5 7B Instruct         (provider: llamacpp, requires server)
    2   - Qwen 2.5 3B Instruct         (provider: llamacpp, requires server)
    3   - Llama 3.1 8B Instruct        (provider: llamacpp, requires server)
    4   - Llama 3.2 3B Instruct        (provider: llamacpp, requires server)
    5   - Mistral 7B Instruct v0.3     (provider: llamacpp, requires server)
    6   - Phi-4 Mini Instruct          (provider: llamacpp, requires server)
    7   - Phi-4 Reasoning              (provider: llamacpp, requires server)
    openai - GPT-4o Mini               (provider: openai, baseline)

Examples:
    python scripts/test_with_llm.py pirate 1
    python scripts/test_with_llm.py inventor openai
    python scripts/test_with_llm.py all 1
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Enable debug mode to see pipeline steps
os.environ["BRIO_DEBUG"] = "1"

from brio_ext.factory import BrioAIFactory  # noqa: E402


# Model configurations matching start_server.sh
MODELS = {
    "1": {
        "name": "qwen-2.5-7b-instruct",
        "display": "Qwen 2.5 7B Instruct",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "2": {
        "name": "qwen-2.5-3b-instruct",
        "display": "Qwen 2.5 3B Instruct",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "3": {
        "name": "llama-3.1-8b-instruct",
        "display": "Llama 3.1 8B Instruct",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "4": {
        "name": "llama-3.2-3b-instruct",
        "display": "Llama 3.2 3B Instruct",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "5": {
        "name": "mistral-7b-instruct-v0.3",
        "display": "Mistral 7B Instruct v0.3",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "6": {
        "name": "phi-4-mini-instruct",
        "display": "Phi-4 Mini Instruct",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "7": {
        "name": "phi-4-reasoning",
        "display": "Phi-4 Reasoning",
        "provider": "llamacpp",
        "base_url": "http://127.0.0.1:8765",
        "requires_server": True,
    },
    "openai": {
        "name": "gpt-4o-mini",
        "display": "GPT-4o Mini",
        "provider": "openai",
        "requires_server": False,
    },
}


def print_separator(char="=", width=80):
    """Print a separator line"""
    print(char * width)


def print_section(title, char="─"):
    """Print a section header"""
    print(f"\n[{title}]")
    print_separator(char, 80)


def load_test_cases():
    """Load test cases from YAML file"""
    test_cases_file = Path(__file__).parent.parent / "fixtures" / "test_cases.yaml"
    with open(test_cases_file) as f:
        return yaml.safe_load(f)


def load_component(relative_path: str) -> str:
    """Load a component file from fixtures directory"""
    if not relative_path:
        return ""
    component_file = Path(__file__).parent.parent / "fixtures" / relative_path
    with open(component_file) as f:
        return f.read().strip()


def assemble_messages(test_case: Dict) -> List[Dict[str, str]]:
    """
    Assemble messages like BrioDocs does:
    1. System message = system prompt + content + insights
    2. User message = user prompt

    For multiturn, build conversation history
    """
    messages = []

    # Handle multi-turn conversation
    if "turns" in test_case:
        # Build system message first
        system_parts = []

        # Load system prompt
        if test_case.get("system"):
            system_parts.append(load_component(test_case["system"]))

        # Add content if present
        if test_case.get("content"):
            system_parts.append("\n\n# CONTENT\n")
            system_parts.append(load_component(test_case["content"]))

        # Add insights if present
        insights = test_case.get("insights", [])
        if insights:
            system_parts.append("\n\n# INSIGHTS\n")
            for insight_path in insights:
                system_parts.append(load_component(insight_path))
                system_parts.append("\n")

        messages.append({
            "role": "system",
            "content": "\n".join(system_parts).strip()
        })

        # Add conversation turns
        for turn in test_case["turns"]:
            # Load user message
            user_content = load_component(turn["user"])
            messages.append({"role": "user", "content": user_content})

            # Add assistant response if provided (for history)
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": turn["assistant"]})

        return messages

    # Single-turn: Assemble system message from components
    system_parts = []

    # Load system prompt
    if test_case.get("system"):
        system_parts.append(load_component(test_case["system"]))

    # Add content if present
    if test_case.get("content"):
        system_parts.append("\n\n# CONTENT\n")
        system_parts.append(load_component(test_case["content"]))

    # Add insights if present
    insights = test_case.get("insights", [])
    if insights:
        system_parts.append("\n\n# INSIGHTS\n")
        for insight_path in insights:
            system_parts.append(load_component(insight_path))
            system_parts.append("\n")

    # Create system message
    messages.append({
        "role": "system",
        "content": "\n".join(system_parts).strip()
    })

    # Add user message
    if test_case.get("user"):
        user_content = load_component(test_case["user"])
        messages.append({"role": "user", "content": user_content})

    return messages


def truncate_text(text: str, max_chars: int = 500, show_both_ends: bool = True) -> str:
    """Truncate long text for display"""
    if len(text) <= max_chars:
        return text

    if show_both_ends and max_chars > 200:
        half = (max_chars - 20) // 2
        return f"{text[:half]}\n\n... [{len(text) - max_chars} chars omitted] ...\n\n{text[-half:]}"
    else:
        return f"{text[:max_chars]}\n\n... [{len(text) - max_chars} chars omitted] ..."


def format_messages(messages: List[Dict[str, str]], truncate: bool = True) -> str:
    """Format messages for display"""
    lines = []
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"]

        if truncate and len(content) > 300:
            content = truncate_text(content, 300, show_both_ends=False)

        lines.append(f"{i}. {role}:")
        lines.append(f"   {content}")
        lines.append("")

    return "\n".join(lines)


def test_model(
    model_config: Dict,
    scenario_name: str,
    test_case: Dict,
) -> bool:
    """Test a single model with a scenario, showing full 4-step pipeline"""

    print_separator()
    print(f"SCENARIO: {scenario_name}")
    print(f"MODEL: {model_config['display']}")
    print(f"PROVIDER: {model_config['provider']}")
    if model_config.get("base_url"):
        print(f"BASE_URL: {model_config['base_url']}")
    print(f"DESCRIPTION: {test_case['description']}")
    print_separator()

    # Assemble messages from components
    messages = assemble_messages(test_case)

    # [STEP 1] TEST → ESPERANTO/BRIO_EXT (Input Messages)
    print_section("STEP 1: TEST → ESPERANTO/BRIO_EXT (Input Messages)")
    print("Messages being sent to brio_ext:")
    print(format_messages(messages))

    # [STEP 2] ESPERANTO/BRIO_EXT → LLM SERVER (Converted/Rendered Prompt)
    print_section("STEP 2: ESPERANTO/BRIO_EXT → LLM SERVER")
    print("(Watch debug output below for rendered prompt sent to model)")
    print("")

    try:
        # Create model instance
        config = {
            "temperature": 0.7,
            "max_tokens": 512,
        }
        if model_config.get("base_url"):
            config["base_url"] = model_config["base_url"]

        model = BrioAIFactory.create_language(
            provider=model_config["provider"],
            model_name=model_config["name"],
            config=config
        )

        # Call the model - BRIO_DEBUG will show the conversion
        print("Calling model...")
        response = model.chat_complete(messages)
        print("")

        # [STEP 3] BRIO_EXT → TEST (Fenced Response)
        print_section("STEP 3: BRIO_EXT → TEST (Fenced Response)")
        content = response.choices[0].message.content
        print("Response with brio_ext fencing applied:")
        print(content)

        # [STEP 4] PARSE & VALIDATE
        print_section("STEP 4: PARSE & VALIDATE")

        # Check for fencing
        has_out_open = "<out>" in content
        has_out_close = "</out>" in content
        properly_fenced = has_out_open and has_out_close

        if properly_fenced:
            start = content.find("<out>") + 5
            end = content.find("</out>")
            parsed_content = content[start:end].strip()
            print("✓ Response properly fenced in <out>...</out>")
            print("\nExtracted content (ready for BrioDocs):")
            print(parsed_content)
        else:
            parsed_content = content
            print("✗ No <out>...</out> fencing found (this shouldn't happen!)")
            print("\nContent:")
            print(parsed_content)

        # [METADATA]
        print_section("RESPONSE METADATA")
        print(f"Model: {model_config['display']}")
        print(f"Provider: {model_config['provider']}")
        print(f"Finish reason: {response.choices[0].finish_reason}")

        if hasattr(response, 'usage') and response.usage:
            print(f"Prompt tokens: {response.usage.prompt_tokens:,}")
            print(f"Completion tokens: {response.usage.completion_tokens:,}")
            print(f"Total tokens: {response.usage.total_tokens:,}")

        # [VALIDATION]
        print_section("VALIDATION")

        validation_passed = True

        # Check fencing
        validation = test_case.get("validation", {})
        if validation.get("should_fence", False):
            if properly_fenced:
                print("✓ Response properly fenced in <out>...</out>")
            else:
                print("✗ Response NOT properly fenced")
                validation_passed = False
        else:
            print("○ Fencing not required for this test")

        # Check for expected content (all must be present)
        should_contain = validation.get("should_contain", [])
        if should_contain:
            print(f"\nChecking for expected content ({len(should_contain)} phrases):")
            for phrase in should_contain:
                if phrase.lower() in parsed_content.lower():
                    print(f"  ✓ Found: '{phrase}'")
                else:
                    print(f"  ✗ Missing: '{phrase}'")
                    validation_passed = False

        # Check for expected content (at least one from each group must be present)
        should_contain_any = validation.get("should_contain_any", [])
        if should_contain_any:
            print(f"\nChecking for expected content ({len(should_contain_any)} groups, any match per group):")
            for group in should_contain_any:
                found = False
                found_phrase = None
                for phrase in group:
                    if phrase.lower() in parsed_content.lower():
                        found = True
                        found_phrase = phrase
                        break

                if found:
                    alternatives = " OR ".join(f"'{p}'" for p in group)
                    print(f"  ✓ Found '{found_phrase}' (from: {alternatives})")
                else:
                    alternatives = " OR ".join(f"'{p}'" for p in group)
                    print(f"  ✗ Missing any of: {alternatives}")
                    validation_passed = False

        # Check stop token
        if properly_fenced:
            if response.choices[0].finish_reason in ["stop", "length"]:
                print(f"\n✓ Stop reason: {response.choices[0].finish_reason}")
            else:
                print(f"\n○ Unusual stop reason: {response.choices[0].finish_reason}")

        # Notes
        notes = validation.get("notes")
        if notes:
            print(f"\nNotes: {notes}")

        # [OVERALL RESULT]
        print_section("OVERALL RESULT")
        if validation_passed:
            print("✅ TEST PASSED")
        else:
            print("❌ TEST FAILED")

        print_separator()
        print()

        return validation_passed

    except Exception as e:
        print_section("ERROR")
        print(f"❌ Exception occurred: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print_separator()
        print()
        return False


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    scenario_arg = sys.argv[1]
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "1"

    # Validate model selection
    if model_arg not in MODELS:
        print(f"Error: Unknown model '{model_arg}'")
        print("\nAvailable models:")
        for key, config in MODELS.items():
            print(f"  {key:8s} - {config['display']}")
        sys.exit(1)

    model_config = MODELS[model_arg]

    # Check prerequisites
    if model_arg == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if model_config.get("requires_server"):
        import httpx
        try:
            response = httpx.get(f"{model_config['base_url']}/v1/models", timeout=2)
            if response.status_code != 200:
                raise Exception("Server not responding")
        except Exception as e:
            print(f"Error: llama.cpp server not running at {model_config['base_url']}")
            print("\nStart the server with the model you want to test:")
            print(f"  ./scripts/start_server.sh")
            print(f"\nThen select model {model_arg} ({model_config['display']})")
            print(f"\nDebug: {e}")
            sys.exit(1)

    # Load test cases
    test_cases = load_test_cases()

    # Determine which scenarios to run
    if scenario_arg == "all":
        scenario_names = list(test_cases.keys())
    elif scenario_arg in test_cases:
        scenario_names = [scenario_arg]
    else:
        print(f"Error: Unknown scenario '{scenario_arg}'")
        print("Available scenarios: " + ", ".join(test_cases.keys()) + ", all")
        sys.exit(1)

    # Run tests
    results = {}

    print("\n\n")
    print("=" * 80)
    print("BRIO_EXT MANUAL TEST RUNNER")
    print("=" * 80)
    print(f"Model: {model_config['display']}")
    print(f"Provider: {model_config['provider']}")
    print(f"Scenarios: {', '.join(scenario_names)}")
    print(f"Debug mode: ENABLED (BRIO_DEBUG=1)")
    print("=" * 80)
    print("\n")

    for scenario_name in scenario_names:
        test_case = test_cases[scenario_name]

        # Test the model
        passed = test_model(
            model_config=model_config,
            scenario_name=scenario_name,
            test_case=test_case,
        )

        results[scenario_name] = passed

        # Pause between scenarios if running multiple
        if len(scenario_names) > 1 and scenario_name != scenario_names[-1]:
            input("\nPress Enter to continue to next scenario...")
            print("\n\n")

    # Final summary
    print("\n\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Model: {model_config['display']}")
    print(f"Provider: {model_config['provider']}")
    print()

    for scenario_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {scenario_name:15s} {status}")

    print("=" * 80)

    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All tests passed!\n")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
