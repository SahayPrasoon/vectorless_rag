"""
src/config/llm.py
─────────────────
LLM provider selection via environment variables.

Set ONE of these in your .env:

  # Use Groq (default)
  LLM_PROVIDER=groq
  GROQ_API_KEY=your_key

  # Use OpenAI
  LLM_PROVIDER=openai
  OPENAI_API_KEY=your_key
  OPENAI_MODEL=gpt-4o          # optional, defaults to gpt-4o
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_provider = os.getenv("LLM_PROVIDER", "groq").lower()

if _provider == "openai":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("LLM initialised: %s via OpenAI.", os.getenv("OPENAI_MODEL", "gpt-4o"))

elif _provider == "groq":
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    logger.info("LLM initialised: %s via Groq.", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

else:
    raise RuntimeError(
        f"Unknown LLM_PROVIDER='{_provider}'. Supported values: groq, openai"
    )