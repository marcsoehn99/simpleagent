import os
import asyncio
from dotenv import load_dotenv


from agents import Agent, Runner, function_tool


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
    model="gpt-5.2"
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())