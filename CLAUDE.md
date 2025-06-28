
Esperanto is a library that helps developers to work with multiple AI models using a simplified and single interface. So, it's very importante that we are consistent across providers and that our documentation is very clear. 

When building a provider, always look at their base class and sibling providers for confirmation.
We want to make the interface as consistent as possible since this is the main value proposition of this project.

If we are adding a new provider, we need to expose it through the AIFactory class. Always look at the base class for the type of provider you are adding, like Language or Embedding and also check a couple implementations for different providers to get a sense of how it should work.

Every time you write new tests, you should test if they are working: `uv run pytest -v`