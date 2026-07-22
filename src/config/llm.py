"""
src/config/llm.py
─────────────────
Single shared LLM instance (Groq / llama-3.3-70b-versatile).
Import `llm` anywhere you need to call the model.
"""
import os
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

logger.info("LLM initialised: llama-3.3-70b-versatile via Groq.")