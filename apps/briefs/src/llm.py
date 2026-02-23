import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required for brief generation.")

    google_base_url = os.environ.get("GOOGLE_BASE_URL", "").strip()
    if not google_base_url:
        google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    _client = OpenAI(
        api_key=google_api_key,
        base_url=google_base_url,
    )
    return _client


def call_llm(model: str, messages: list[dict], temperature: float = 0):
    if not model:
        raise ValueError("Model name is required.")
    if not messages:
        raise ValueError("At least one message is required.")

    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=temperature,
    )

    return response.choices[0].message.content, (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

    # return {
    #     "answer": response.choices[0].message.content,
    #     "usage": {
    #         "input_tokens": response.usage.prompt_tokens,
    #         "output_tokens": response.usage.completion_tokens,
    #     },
    # }
