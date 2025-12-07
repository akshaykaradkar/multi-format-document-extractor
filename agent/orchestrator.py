"""
OpenAI Function Calling Agent for Document Processing

This orchestrator uses OpenAI's function calling to interact with
the document processing tools through natural language.

Author: Akshay Karadkar
"""

import json
import sys
from pathlib import Path
from typing import Optional, Generator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from mcp_server.tools import DocumentTools
from .prompts import SYSTEM_PROMPT, get_openai_tools


class DocumentAgent:
    """
    LLM-powered agent for document processing using OpenAI function calling.

    Provides a natural language interface to the document processing pipeline,
    allowing users to process documents, compare modes, and analyze results
    through conversation.
    """

    def __init__(
        self,
        model: str = None,
        verbose: bool = True,
        max_iterations: int = 10
    ):
        """
        Initialize the document processing agent.

        Args:
            model: OpenAI model to use (defaults to config)
            verbose: Print processing steps
            max_iterations: Maximum tool calls per request
        """
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY in .env file."
            )

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or OPENAI_MODEL
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Initialize tools
        self.tools = DocumentTools(verbose=False)
        self.openai_tools = get_openai_tools()

        # Conversation history
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def process(self, user_input: str) -> str:
        """
        Process a natural language request.

        Args:
            user_input: User's request in natural language

        Returns:
            Agent's response as string
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User: {user_input}")
            print(f"{'='*60}")

        # Run the agent loop
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.openai_tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # If no tool calls, we have the final response
            if not message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                if self.verbose:
                    print(f"\nAgent: {message.content}")
                return message.content

            # Process tool calls
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if self.verbose:
                    print(f"\n[Tool Call] {function_name}")
                    print(f"  Args: {arguments}")

                # Execute the tool
                result = self._execute_tool(function_name, arguments)

                if self.verbose:
                    # Truncate long results for display
                    display_result = result[:500] + "..." if len(result) > 500 else result
                    print(f"  Result: {display_result}")

                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # If we hit max iterations, return a fallback response
        return "I've reached the maximum number of operations. Please try a simpler request."

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result as string."""
        try:
            if name == "process_document":
                result = self.tools.process_document(
                    arguments.get("file_path", ""),
                    arguments.get("mode", "hybrid")
                )
            elif name == "list_sample_files":
                result = self.tools.list_sample_files()
            elif name == "get_confidence_report":
                result = self.tools.get_confidence_report(
                    arguments.get("file_path", "")
                )
            elif name == "compare_extraction_modes":
                result = self.tools.compare_extraction_modes(
                    arguments.get("file_path", "")
                )
            elif name == "get_extraction_stats":
                file_paths = arguments.get("file_paths")
                if file_paths:
                    file_paths = [p.strip() for p in file_paths.split(",")]
                result = self.tools.get_extraction_stats(file_paths)
            else:
                result = {"error": f"Unknown tool: {name}"}

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Stream the agent's response.

        Args:
            user_input: User's request

        Yields:
            Response chunks as they become available
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            # First, get tool calls without streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.openai_tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # If there are tool calls, execute them
            if message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                for tool_call in message.tool_calls:
                    yield f"\n[Calling {tool_call.function.name}...]\n"

                    result = self._execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments)
                    )

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                continue

            # No tool calls - stream the final response
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            self.messages.append({
                "role": "assistant",
                "content": full_response
            })
            return

    def reset(self):
        """Reset conversation history."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def get_history(self) -> list:
        """Get conversation history."""
        return self.messages.copy()


# Interactive mode for direct testing
if __name__ == "__main__":
    print("Document Processing Agent")
    print("=" * 60)
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 60)

    try:
        agent = DocumentAgent(verbose=True)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation reset.")
                continue

            response = agent.process(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
