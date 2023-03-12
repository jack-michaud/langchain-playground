from langchain.agents import initialize_agent
from tools import add_memory, search_memory
from langchain.llms import OpenAIChat

from dotenv import load_dotenv

load_dotenv()

# GPT-3.5
llm = OpenAIChat()

agent = initialize_agent(
    [
        add_memory,
        search_memory
    ],
    llm,
    agent="zero-shot-react-description", 
    verbose=True,
    agent_kwargs={
        "prefix": "You are searching for underlying meaning in your memory. You have access to the following tools:",
    },
)

if __name__ == "__main__":
    while True:
        i = input("Begin rumination")
        print(agent(i))

