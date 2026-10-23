# import os

# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

# load_dotenv()


# def get_gemini():
#     return ChatGoogleGenerativeAI(
#         model="gemini-3.8-flash",
#         temperature=0.2,
#         google_api_key=os.getenv("GOOGLE_API_KEY"),
#     )


# def get_groq():
#     return ChatGroq(
#         model="openai/gpt-oss-120b",
#         temperature=0.2,
#         max_tokens=1200,
#         groq_api_key=os.getenv("GROQ_API_KEY"),
#     )


# def get_llm(provider=None):
#     provider = provider or os.getenv("LLM_PROVIDER", "gemini")

#     if provider == "gemini":
#         return get_gemini()

#     if provider == "groq":
#         return get_groq()

#     raise ValueError(f"Unknown provider: {provider}")

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MAX_TOKENS = 1200


def get_gemini(max_tokens: int = DEFAULT_MAX_TOKENS):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=max_tokens,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_groq(max_tokens: int = DEFAULT_MAX_TOKENS, reasoning_effort: str = "low"):
    # openai/gpt-oss-120b is a reasoning model: it spends tokens on a hidden
    # reasoning trace before writing the visible answer. If that trace is
    # long, it can consume the whole max_tokens budget before any visible
    # text is written, leaving response.content empty no matter how large
    # max_tokens is. reasoning_effort caps how much hidden reasoning it does,
    # which both fixes that failure mode and reduces token usage on every
    # call, not just the ones that were coming back empty.
    #
    # NOTE: if your installed langchain_groq version rejects this as an
    # unexpected keyword argument, move it into model_kwargs instead:
    #   ChatGroq(..., model_kwargs={"reasoning_effort": reasoning_effort})
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def get_llm(provider=None, max_tokens: int = DEFAULT_MAX_TOKENS, reasoning_effort: str = "low"):
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        return get_gemini(max_tokens=max_tokens)

    if provider == "groq":
        return get_groq(max_tokens=max_tokens, reasoning_effort=reasoning_effort)

    raise ValueError(f"Unknown provider: {provider}")