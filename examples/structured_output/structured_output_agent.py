"""StructuredOutputAgent: validate the answer and ask the model to repair it if it is wrong.

The agent runs GENERATE, checks the reply against your schema, and on failure injects
the validation error plus the target JSON Schema and tries again, up to `max_attempts`
repairs per turn. Tools can be used inside the same loop before the final answer.

Run: python examples/structured_output/structured_output_agent.py   (needs OPENAI_API_KEY)
"""

import asyncio
import json
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import StructuredOutputAgent


load_dotenv()


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str
    humidity_percent: int
    advice: Literal["umbrella", "sunscreen", "jacket", "none"]


def get_current_weather(city: str) -> str:
    """Return raw weather data for a city as JSON."""
    return json.dumps({"city": city, "temp_c": 22.5, "condition": "partly cloudy", "humidity": 68})


async def main() -> None:
    agent = StructuredOutputAgent(
        model="gpt-5-mini",
        output_schema=WeatherReport,
        tools=[get_current_weather],
        system_prompt=[
            {
                "role": "system",
                "content": "Fetch the weather with the tool, then answer with a WeatherReport.",
            }
        ],
        # Repair attempts per turn. After that, the last reply is returned as it is,
        # so check validity before trusting it (done below).
        max_attempts=3,
    )
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("What is the weather in Paris today?")]},
        config={"thread_id": "weather-1"},
    )
    last = result["messages"][-1]

    try:
        report = WeatherReport.model_validate_json(last.text())
    except ValueError as exc:
        print("model never produced valid output:", exc)
    else:
        print(f"{report.city}: {report.temperature_celsius} C, {report.condition}")
        print(f"humidity {report.humidity_percent}%, advice: {report.advice}")

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
