"""
OpenAI Agent for Document Processing

Provides an LLM-powered agent that uses function calling to interact
with the MCP document processing tools.

Author: Akshay Karadkar
"""

from .orchestrator import DocumentAgent

__all__ = ["DocumentAgent"]
