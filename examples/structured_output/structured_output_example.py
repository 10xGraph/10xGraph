"""StructuredOutputAgent — runnable example.

Demonstrates two scenarios:
  1. Simple structured extraction without tools.
  2. Structured extraction with a helper tool (live weather fetch).

Run with a valid OPENAI_API_KEY (or any OpenAI-compatible key):

    python examples/structured_output/structured_output_example.py
"""

import asyncio
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tenxgraph.prebuilt.agent import StructuredOutputAgent
from tenxgraph.storage.checkpointer import InMemoryCheckpointer


load_dotenv()

# ---------------------------------------------------------------------------
# Example 1 — Simple structured extraction
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    """Expected output schema for movie review extraction."""

    title: str = Field(description="Movie title")
    director: str = Field(description="Director's name")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    summary: str = Field(description="One-sentence summary of the movie")


async def example_simple() -> None:
    print("\n" + "=" * 60)
    print("Example 1: Structured movie-review extraction")
    print("=" * 60)

    agent = StructuredOutputAgent(
        model="gpt-4o-mini",
        output_schema=MovieReview,
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You are a film critic. Extract structured information about movies when asked."
                ),
            }
        ],
        max_attempts=3,
        provider="openai",
    )

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    result = await app.ainvoke(
        {"message": "Give me a review of The Matrix (1999)."},
        config={"thread_id": "movie-review-1"},
    )

    last_msg = result["context"][-1]
    print(f"Raw text  : {last_msg.text()[:200]}")
    print(f"parsed_content: {last_msg.parsed_content}")

    # The parsed_content should be a MovieReview instance (when using
    # OpenAI's structured-output parse path).  Fall back to JSON text.
    if isinstance(last_msg.parsed_content, MovieReview):
        review: MovieReview = last_msg.parsed_content
    else:
        data = json.loads(last_msg.text())
        review = MovieReview(**data)

    print(f"\nTitle    : {review.title}")
    print(f"Director : {review.director}")
    print(f"Year     : {review.year}")
    print(f"Rating   : {review.rating}/10")
    print(f"Summary  : {review.summary}")


# ---------------------------------------------------------------------------
# Example 2 — Structured extraction with a tool
# ---------------------------------------------------------------------------


class WeatherReport(BaseModel):
    """Structured weather report."""

    city: str
    temperature_celsius: float
    condition: str
    humidity_percent: int
    advice: str = Field(description="One short piece of advice for the day")


def get_current_weather(city: str) -> str:
    """Simulate a weather API call. Returns a raw JSON string."""
    # Simulated data — replace with a real API call in production.
    mock_data = {
        "city": city,
        "temp_c": 22.5,
        "condition": "partly cloudy",
        "humidity": 68,
    }
    return json.dumps(mock_data)


async def example_with_tools() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Structured output + tool use")
    print("=" * 60)

    agent = StructuredOutputAgent(
        model="gpt-4o-mini",
        output_schema=WeatherReport,
        tools=[get_current_weather],
        system_prompt=[
            {
                "role": "system",
                "content": (
                    "You are a weather assistant. Use the provided tool to fetch "
                    "weather data, then return a structured WeatherReport."
                ),
            }
        ],
        max_attempts=2,
        provider="openai",
    )

    app = agent.compile(checkpointer=InMemoryCheckpointer())

    result = await app.ainvoke(
        {"message": "What's the weather like in Paris today?"},
        config={"thread_id": "weather-1"},
    )

    last_msg = result["context"][-1]

    if isinstance(last_msg.parsed_content, WeatherReport):
        report: WeatherReport = last_msg.parsed_content
    else:
        data = json.loads(last_msg.text())
        report = WeatherReport(**data)

    print(f"City        : {report.city}")
    print(f"Temperature : {report.temperature_celsius} °C")
    print(f"Condition   : {report.condition}")
    print(f"Humidity    : {report.humidity_percent}%")
    print(f"Advice      : {report.advice}")


# ---------------------------------------------------------------------------
# Example 3 — TypedDict schema
# ---------------------------------------------------------------------------
from typing import TypedDict  # noqa: E402


class PersonInfo(TypedDict):
    name: str
    age: int
    occupation: str


async def example_typeddict() -> None:
    print("\n" + "=" * 60)
    print("Example 3: TypedDict output schema")
    print("=" * 60)

    agent = StructuredOutputAgent(
        model="gpt-4o-mini",
        output_schema=PersonInfo,
        system_prompt=[
            {
                "role": "system",
                "content": "Extract person information as structured data.",
            }
        ],
        max_attempts=2,
        provider="openai",
    )

    app = agent.compile()

    result = await app.ainvoke(
        {"message": "Alice is a 30-year-old software engineer."},
        config={"thread_id": "person-1"},
    )

    last_msg = result["context"][-1]
    raw = last_msg.text()
    info: PersonInfo = json.loads(raw)  # type: ignore[assignment]
    print(f"Name       : {info['name']}")
    print(f"Age        : {info['age']}")
    print(f"Occupation : {info['occupation']}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(example_simple())
    asyncio.run(example_with_tools())
    asyncio.run(example_typeddict())
