"""
MCP Server for Document Processing

Provides MCP (Model Context Protocol) interface to the HybridPipeline.
This enables AI agents to interact with the document processing system
using standardized tool calling patterns.

Author: Akshay Karadkar
"""

from .tools import DocumentTools
from .server import mcp, run_server

__all__ = ["DocumentTools", "mcp", "run_server"]
