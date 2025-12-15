"""
NASEA Web Interface
Built with Chainlit for real-time chat + tool display
"""

import chainlit as cl
from chainlit.input_widget import Select, TextInput
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nasea.llm.llm_factory import LLMFactory
from nasea.core.tool_definitions import GENERATION_TOOLS
from nasea.core.tool_executors import ToolExecutor
from nasea.prompts import load_system_prompt
from pathlib import Path
import json


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    # Create output directory for this session
    session_id = cl.user_session.get("id")
    output_dir = Path(f"output/web_{session_id[:8]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store in session
    cl.user_session.set("output_dir", str(output_dir))
    cl.user_session.set("messages", [])

    # Get LLM client
    try:
        client = LLMFactory.create_client()
        cl.user_session.set("client", client)
        model_name = getattr(client, 'model', 'Unknown')
    except Exception as e:
        await cl.Message(content=f"Error initializing: {e}\n\nMake sure API keys are set in .env").send()
        return

    # Welcome message
    await cl.Message(
        content=f"""# Welcome to NASEA Web

I'm your AI coding assistant. I can:
- **Create** projects from descriptions
- **Edit** existing code
- **Debug** issues
- **Search** the web for info

**Model:** {model_name}

What would you like to build today?"""
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    client = cl.user_session.get("client")
    if not client:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    output_dir = Path(cl.user_session.get("output_dir"))
    messages = cl.user_session.get("messages", [])

    # Add user message
    messages.append({"role": "user", "content": message.content})

    # Check if this is a creation request
    is_creation = any(word in message.content.lower() for word in
                      ["create", "build", "make", "generate", "write"])

    if is_creation:
        await handle_creation(client, messages, output_dir, message.content)
    else:
        await handle_chat(client, messages)


async def handle_chat(client, messages):
    """Handle simple chat messages."""
    # Show thinking
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Get response
        response = client.chat(
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )

        text = response.choices[0].message.content
        msg.content = text
        await msg.update()

        # Save to history
        messages.append({"role": "assistant", "content": text})
        cl.user_session.set("messages", messages)

    except Exception as e:
        msg.content = f"Error: {e}"
        await msg.update()


async def handle_creation(client, messages, output_dir, user_prompt):
    """Handle project creation with tool use."""
    # Create tool executor
    executor = ToolExecutor(output_dir)

    # Load system prompt
    system_prompt = load_system_prompt("create")
    system_prompt = system_prompt.replace("${PROJECT_ROOT}", str(output_dir))

    # Build messages with system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Create initial message
    msg = cl.Message(content="")
    await msg.send()

    steps_content = ""
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        try:
            # Call LLM with tools
            response = client.chat_with_tools(
                messages=full_messages,
                tools=GENERATION_TOOLS,
                max_tokens=4096,
                temperature=0.7
            )

            assistant_message = response.choices[0].message

            # Check for text content
            if assistant_message.content:
                steps_content += assistant_message.content + "\n\n"
                msg.content = steps_content
                await msg.update()

            # Check for tool calls
            tool_calls = getattr(assistant_message, 'tool_calls', None)
            if not tool_calls:
                break

            # Add assistant message to history
            full_messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # Execute each tool
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except:
                    args = {}

                # Show tool step
                step = cl.Step(name=func_name, type="tool")
                await step.send()

                # Execute tool
                result = executor.execute(func_name, args)

                # Update step with result
                if result.get("success"):
                    step.output = result.get("message", "Success")
                    if func_name == "write_file":
                        file_path = args.get("file_path", "file")
                        lines = result.get("line_count", 0)
                        step.output = f"Created {file_path} ({lines} lines)"
                else:
                    step.output = f"Error: {result.get('error', 'Unknown error')}"

                await step.update()

                # Add tool result to messages
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                # Check if complete
                if func_name == "complete_generation":
                    # Show completion summary
                    summary = result.get("summary", "Project created!")
                    steps_content += f"\n\n**{summary}**\n"
                    steps_content += f"\nFiles saved to: `{output_dir}`"
                    msg.content = steps_content
                    await msg.update()

                    # Save to session history
                    messages.append({"role": "assistant", "content": steps_content})
                    cl.user_session.set("messages", messages)
                    return

        except Exception as e:
            steps_content += f"\n\nError: {e}"
            msg.content = steps_content
            await msg.update()
            break

    # Save final state
    messages.append({"role": "assistant", "content": steps_content})
    cl.user_session.set("messages", messages)


if __name__ == "__main__":
    cl.run()
