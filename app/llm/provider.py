import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()


def get_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_groq():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=1200,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def get_llm(provider=None):
    provider = provider or os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        return get_gemini()

    if provider == "groq":
        return get_groq()

    raise ValueError(f"Unknown provider: {provider}")