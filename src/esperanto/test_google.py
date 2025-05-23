import asyncio

from esperanto import GoogleLanguageModel

messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
    ]


params = {
    "max_tokens": 850,
    "temperature": 1.0,
    "streaming": False,
    "top_p": 0.9,
    "structured": None
}

models = {
    "google-flash": {"class": GoogleLanguageModel, "model": "gemini-2.0-flash"},
    "google-pro": {"class": GoogleLanguageModel, "model": "gemini-2.5-pro-preview-05-06"},
}


async def main():
    for name, config in models.items():
        try:
            # Create an instance of the provider class
            provider = config["class"]()
            print(f"\n=== {name.upper()} Models ===")
            print(provider.models)
        except Exception as e:
            print(f"Failed to get models for {name}: {e}")
    
    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            result = llm.chat_complete(messages)
            print(result.choices[0].message.content)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to get models for {name}: {e}")


    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            result = await llm.achat_complete(messages)
            print(result.choices[0].message.content)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to get models for {name}: {e}")

    import json

    json_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Please return the top 3 brazilian cities in JSON format"},
        ]


    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"], structured={"type": "json"})
            print(f"Results for {llm.provider}:")
            result = llm.chat_complete(json_messages)
            try:
                json_data = json.loads(result.choices[0].message.content)
                print(json_data)
            except json.JSONDecodeError:
                print("Error decoding JSON")
                print(result.choices[0].message.content)
            
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to get models for {name}: {e}")



    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            result = llm.chat_complete(
                messages, stream=True
            )

            for chunk in result:
                print(chunk)
            print("\n" + "="*50 + "\n")
        except Exception as e:
                print(f"Failed to process for {name}: {e}")




    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            result = await llm.achat_complete(
                messages, stream=True
            )

            async for chunk in result:
                print(chunk)
            print("\n" + "="*50 + "\n")
        except Exception as e:
                    print(f"Failed to process for {name}: {e}")



    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            model = llm.to_langchain()
            response = model.invoke(messages)
            print(response.content)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to process for {name}: {e}")



    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"])
            print(f"Results for {llm.provider}:")
            model = llm.to_langchain()
            response = await model.ainvoke(messages)
            print(response.content)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to process for {name}: {e}")


    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"], streaming=True)
            print(f"Results for {llm.provider}:")
            model = llm.to_langchain()
            response = model.stream(messages)
            for chunk in response:
                print(chunk)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to process for {name}: {e}")


    for name, config in models.items():
        try:
            llm = config["class"](model_name=config["model"], streaming=True)
            print(f"Results for {llm.provider}:")
            model = llm.to_langchain()
            response = model.astream(messages)
            async for chunk in response:
                print(chunk)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Failed to process for {name}: {e}")

            
if __name__ == "__main__":
    asyncio.run(main())