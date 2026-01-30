"""
Pydantic AI Integration Example

This example demonstrates how to use Esperanto with Pydantic AI agents.
The key benefit is provider flexibility - the same agent code works with
any of Esperanto's 15+ supported providers.

Requirements:
    pip install esperanto pydantic-ai

Usage:
    python pydantic_ai_example.py
"""

import asyncio
import random
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from esperanto import AIFactory


# =============================================================================
# Example 1: Basic Agent
# =============================================================================

async def basic_agent_example():
    """Simple agent that answers questions."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Agent")
    print("=" * 60)

    # Create Esperanto model and convert to Pydantic AI
    model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()

    # Create agent
    agent = Agent(model)

    # Run the agent
    result = await agent.run("What is the capital of Japan?")
    print(f"Response: {result.output}")


# =============================================================================
# Example 2: Agent with Tools
# =============================================================================

async def tool_calling_example():
    """Agent that can use tools to perform actions."""
    print("\n" + "=" * 60)
    print("Example 2: Agent with Tools")
    print("=" * 60)

    model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()

    agent = Agent(
        model,
        deps_type=str,  # Dependencies type (player name)
        instructions=(
            "You're a dice game. Roll the die and check if the number "
            "matches the user's guess. Use the player's name in your response."
        ),
    )

    @agent.tool_plain
    def roll_dice() -> str:
        """Roll a six-sided die and return the result."""
        result = random.randint(1, 6)
        print(f"  [Tool called: roll_dice() -> {result}]")
        return str(result)

    @agent.tool
    def get_player_name(ctx: RunContext[str]) -> str:
        """Get the player's name."""
        print(f"  [Tool called: get_player_name() -> {ctx.deps}]")
        return ctx.deps

    result = await agent.run("My guess is 4", deps="Alice")
    print(f"Response: {result.output}")


# =============================================================================
# Example 3: Structured Output
# =============================================================================

class MovieReview(BaseModel):
    """Structured output for movie reviews."""
    title: str
    year: Optional[int] = None
    rating: int  # 1-10
    summary: str
    recommendation: bool


async def structured_output_example():
    """Agent that returns structured data using Pydantic models."""
    print("\n" + "=" * 60)
    print("Example 3: Structured Output")
    print("=" * 60)

    model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()

    agent = Agent(model, output_type=MovieReview)

    result = await agent.run("Review the movie 'The Matrix'")
    review = result.output

    print(f"Title: {review.title}")
    print(f"Year: {review.year}")
    print(f"Rating: {review.rating}/10")
    print(f"Summary: {review.summary}")
    print(f"Recommended: {'Yes' if review.recommendation else 'No'}")


# =============================================================================
# Example 4: Provider Switching
# =============================================================================

async def provider_switching_example():
    """Demonstrate switching providers without changing agent code."""
    print("\n" + "=" * 60)
    print("Example 4: Provider Switching")
    print("=" * 60)

    # Same agent creation pattern, different providers
    providers = [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-20250514"),
        # Add more providers as needed:
        # ("google", "gemini-2.5-flash"),
        # ("groq", "llama-3.3-70b-versatile"),
    ]

    for provider, model_name in providers:
        try:
            model = AIFactory.create_language(provider, model_name)
            agent = Agent(model.to_pydantic_ai())

            result = await agent.run("Say hello in one sentence.")
            print(f"{provider}/{model_name}: {result.output}")
        except Exception as e:
            print(f"{provider}/{model_name}: Error - {e}")


# =============================================================================
# Example 5: Streaming Response
# =============================================================================

async def streaming_example():
    """Stream responses for real-time output."""
    print("\n" + "=" * 60)
    print("Example 5: Streaming Response")
    print("=" * 60)

    model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()
    agent = Agent(model)

    print("Streaming: ", end="", flush=True)
    async with agent.run_stream("Count from 1 to 5, with a brief pause between each number.") as response:
        async for chunk in response.stream_text():
            print(chunk, end="", flush=True)
    print()  # Newline after streaming


# =============================================================================
# Example 6: Multi-turn Conversation
# =============================================================================

async def conversation_example():
    """Multi-turn conversation with message history."""
    print("\n" + "=" * 60)
    print("Example 6: Multi-turn Conversation")
    print("=" * 60)

    model = AIFactory.create_language("openai", "gpt-4o").to_pydantic_ai()
    agent = Agent(model)

    # First turn
    result1 = await agent.run("My name is Bob and I like pizza.")
    print(f"User: My name is Bob and I like pizza.")
    print(f"Assistant: {result1.output}")

    # Second turn - continues conversation
    result2 = await agent.run(
        "What's my name and what food do I like?",
        message_history=result1.all_messages()
    )
    print(f"\nUser: What's my name and what food do I like?")
    print(f"Assistant: {result2.output}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("Esperanto + Pydantic AI Integration Examples")
    print("=" * 60)

    # Run examples (comment out any you don't want to run)
    await basic_agent_example()
    await tool_calling_example()
    await structured_output_example()
    # await provider_switching_example()  # Requires multiple API keys
    # await streaming_example()
    # await conversation_example()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
