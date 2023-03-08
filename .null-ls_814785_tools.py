from langchain.agents import tool
import subprocess


@tool("ask_for_approval")
def ask_for_approval(query: str) -> str:
    """Asks Jack for feedback on a response. Rephrase the response if a question is required."""
    # Uses zenity to ask for feedback
    response = subprocess.check_output(["zenity", "--entry", "--text", query])

    return f'Jack said: {response.decode("utf-8").strip()}'


@tool("send_message")
def send_message(message: str) -> str:
    """Send a message after approval. All messages must be approved first."""
    message = message.strip()
    subprocess.check_output(["notify-send", "Copy to clipboard", message])
    subprocess.check_output(["wl-copy", message])
    return f'Message sent: "{message}"'
