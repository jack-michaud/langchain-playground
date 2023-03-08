from dotenv import load_dotenv
from langchain.chains.conversation.memory import \
    ConversationSummaryBufferMemory

load_dotenv()
import os

from langchain.agents import initialize_agent, tool
from langchain.llms import OpenAI, OpenAIChat

# ChatGPT
chatllm = OpenAIChat()
# GPT-3.5
llm = OpenAI()


@tool("rephrase")
def filter(text) -> str:
    """Rephrase for politeness."""
    return chatllm(
        f"Rephrase the following sentence with politeness and curiosity: {text}"
    )


agent = initialize_agent(
    [
        filter,
    ],
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=60,
    ),
    agent_kwargs={
        "prefix": "You are a friendly chatbot. Answers your responses to humans with humbleness.",
    },
)

while True:
    text = input("You: ")
    print("Bot:", agent(f"Human: {text}")["output"])
