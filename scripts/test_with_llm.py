#!/usr/bin/env python3
"""
Simple manual test runner for brio_ext validation with full pipeline visibility.

Usage:
    python scripts/test_with_llm.py [scenario] [model]

Scenarios:
    pirate      - Simple system message test (pirate personality)
    inventor    - Medium context inventor lookup (tests Qwen system message bug)
    multiturn   - Multi-turn conversation
    all         - Run all scenarios

Models:
    qwen-2.5-7b-instruct         (provider: llamacpp, requires server)
    gpt-4o-mini                  (provider: openai, baseline)

Examples:
    python scripts/test_with_llm.py pirate qwen-2.5-7b-instruct
    python scripts/test_with_llm.py inventor gpt-4o-mini
    python scripts/test_with_llm.py all qwen-2.5-7b-instruct
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Enable debug mode to see pipeline steps
os.environ["BRIO_DEBUG"] = "1"

from brio_ext.factory import BrioAIFactory  # noqa: E402


def print_separator(char="=", width=80):
    """Print a separator line"""
    print(char * width)


def print_section(title, char="─"):
    """Print a section header"""
    print(f"\n[{title}]")
    print_separator(char, 80)


def load_scenarios():
    """Load test scenarios from JSON file"""
    scenarios_file = Path(__file__).parent.parent / "fixtures" / "scenarios.json"
    with open(scenarios_file) as f:
        return json.load(f)


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
    model_name: str,
    provider: str,
    scenario_name: str,
    scenario: Dict,
    base_url: Optional[str] = None,
) -> bool:
    """Test a single model with a scenario, showing full pipeline"""

    print_separator()
    print(f"SCENARIO: {scenario_name}")
    print(f"MODEL: {model_name}")
    print(f"PROVIDER: {provider}")
    if base_url:
        print(f"BASE_URL: {base_url}")
    print(f"DESCRIPTION: {scenario['description']}")
    print_separator()

    # [1] INPUT
    print_section("1. INPUT TO BRIO_EXT")
    print("Messages being sent:")
    print(format_messages(scenario["messages"]))

    # [2] BRIO_EXT PROCESSING - Debug output will appear here
    print_section("2. BRIO_EXT PROCESSING")
    print("(Watch for debug output from renderer and adapters)")
    print("")

    try:
        # Create model instance
        config = {
            "temperature": 0.7,
            "max_tokens": 512,
        }
        if base_url:
            config["base_url"] = base_url

        model = BrioAIFactory.create_language(
            provider=provider,
            model_name=model_name,
            config=config
        )

        # [3] CALLING MODEL
        print_section("3. CALLING MODEL")
        print("Making request...")
        print("")

        # Call the model - debug output will show what's sent
        response = model.chat_complete(scenario["messages"])

        # [4] RAW RESPONSE
        print_section("4. RAW RESPONSE FROM PROVIDER")
        content = response.choices[0].message.content
        print(content)

        # [5] METADATA
        print_section("5. RESPONSE METADATA")
        print(f"Model: {model_name}")
        print(f"Provider: {provider}")
        print(f"Finish reason: {response.choices[0].finish_reason}")

        if hasattr(response, 'usage') and response.usage:
            print(f"Prompt tokens: {response.usage.prompt_tokens:,}")
            print(f"Completion tokens: {response.usage.completion_tokens:,}")
            print(f"Total tokens: {response.usage.total_tokens:,}")

        # [6] PARSED CONTENT
        print_section("6. PARSED CONTENT")

        # Check for fencing
        has_out_open = "<out>" in content
        has_out_close = "</out>" in content
        properly_fenced = has_out_open and has_out_close

        if properly_fenced:
            start = content.find("<out>") + 5
            end = content.find("</out>")
            parsed_content = content[start:end].strip()
            print("Extracted from <out>...</out> tags:")
            print(parsed_content)
        else:
            parsed_content = content
            print("(No <out>...</out> fencing found)")
            print(parsed_content)

        # [7] VALIDATION
        print_section("7. VALIDATION")

        validation_passed = True

        # Check fencing
        if scenario["validation"].get("should_fence", False):
            if properly_fenced:
                print("✓ Response properly fenced in <out>...</out>")
            else:
                print("✗ Response NOT properly fenced")
                validation_passed = False
        else:
            print("○ Fencing not required for this test")

        # Check for expected content
        should_contain = scenario["validation"].get("should_contain", [])
        if should_contain:
            print(f"\nChecking for expected content ({len(should_contain)} phrases):")
            for phrase in should_contain:
                if phrase.lower() in parsed_content.lower():
                    print(f"  ✓ Found: '{phrase}'")
                else:
                    print(f"  ✗ Missing: '{phrase}'")
                    validation_passed = False

        # Check stop token
        if properly_fenced:
            if response.choices[0].finish_reason in ["stop", "length"]:
                print(f"\n✓ Stop reason: {response.choices[0].finish_reason}")
            else:
                print(f"\n○ Unusual stop reason: {response.choices[0].finish_reason}")

        # Notes
        notes = scenario["validation"].get("notes")
        if notes:
            print(f"\nNotes: {notes}")

        # [8] OVERALL RESULT
        print_section("8. OVERALL RESULT")
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
    model_arg = sys.argv[2] if len(sys.argv) > 2 else "qwen-2.5-7b-instruct"

    # Model configuration
    models_config = {
        "qwen-2.5-7b-instruct": {
            "provider": "llamacpp",
            "base_url": "http://127.0.0.1:8765",
            "requires_server": True,
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "requires_server": False,
        },
    }

    if model_arg not in models_config:
        print(f"Error: Unknown model '{model_arg}'")
        print("Available models: " + ", ".join(models_config.keys()))
        sys.exit(1)

    model_config = models_config[model_arg]

    # Check prerequisites
    if model_arg == "gpt-4o-mini" and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if model_config.get("requires_server"):
        import httpx
        try:
            # Try /v1/models endpoint instead of /health
            response = httpx.get(f"{model_config['base_url']}/v1/models", timeout=2)
            if response.status_code != 200:
                raise Exception("Server not responding")
        except Exception as e:
            print(f"Error: llama.cpp server not running at {model_config['base_url']}")
            print("\nStart the server first:")
            print(f"  ./scripts/start_server.sh {model_arg}")
            print(f"\nDebug: {e}")
            sys.exit(1)

    # Load scenarios
    scenarios = load_scenarios()

    # Determine which scenarios to run
    if scenario_arg == "all":
        scenario_names = list(scenarios.keys())
    elif scenario_arg in scenarios:
        scenario_names = [scenario_arg]
    else:
        print(f"Error: Unknown scenario '{scenario_arg}'")
        print("Available scenarios: " + ", ".join(scenarios.keys()) + ", all")
        sys.exit(1)

    # Run tests
    results = {}

    print("\n\n")
    print("=" * 80)
    print("BRIO_EXT MANUAL TEST RUNNER")
    print("=" * 80)
    print(f"Model: {model_arg}")
    print(f"Provider: {model_config['provider']}")
    print(f"Scenarios: {', '.join(scenario_names)}")
    print(f"Debug mode: ENABLED (BRIO_DEBUG=1)")
    print("=" * 80)
    print("\n")

    for scenario_name in scenario_names:
        scenario = scenarios[scenario_name]

        # Test the model
        passed = test_model(
            model_name=model_arg,
            provider=model_config["provider"],
            scenario_name=scenario_name,
            scenario=scenario,
            base_url=model_config.get("base_url"),
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
    print(f"Model: {model_arg}")
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
