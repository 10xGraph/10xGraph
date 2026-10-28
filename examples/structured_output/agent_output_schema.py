"""Structured output with output_schema: get a typed Pydantic object back from an agent.

Pass a Pydantic model as `output_schema` and the provider is asked to answer in that
shape. The parsed object is on `Message.parsed_content`; the JSON text is also in the
message. ReactAgent forwards `output_schema` to its inner Agent.

Run: python examples/structured_output/agent_output_schema.py   (needs OPENAI_API_KEY)
"""

import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    director: str = Field(description="Director's name")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    summary: str = Field(description="One-sentence summary")


async def main() -> None:
    # output_schema only applies to text output. The same Agent(output_schema=...) argument
    # works when you build your own StateGraph.
    agent = ReactAgent(
        model="gpt-5-mini",
        output_schema=MovieReview,
        system_prompt=[{"role": "system", "content": "You are a film critic."}],
    )
    app = agent.compile()

    result = await app.ainvoke(
        {"messages": [Message.text_message("Review The Matrix (1999).")]},
        config={"thread_id": "review-1"},
    )
    last = result["messages"][-1]

    # parsed_content is the validated object when the provider returns one. Fall back to
    # validating the JSON text so the example works across providers.
    if isinstance(last.parsed_content, MovieReview):
        review = last.parsed_content
    else:
        review = MovieReview.model_validate_json(last.text())

    print(f"{review.title} ({review.year}) directed by {review.director}")
    print(f"rating: {review.rating}/10")
    print(f"summary: {review.summary}")

    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
