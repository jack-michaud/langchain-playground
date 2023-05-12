from dotenv import load_dotenv
from langchain.agents.mrkl.prompt import PREFIX, SUFFIX
from langchain.chains.conversation.memory import \
    ConversationSummaryBufferMemory

load_dotenv()

from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI, OpenAIChat

from tools import add_memory, ask_for_approval, search_memory, send_message

llm = OpenAIChat()

agent = initialize_agent(
    [
        add_memory,
        search_memory,
    ],
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={
        "prefix": "You are a friendly assistant responsible for a knowledgebase. You work with a human to help them remember things."
        + PREFIX,
        "suffix": "If you do not have an answer for something from memory, respond with 'Final Answer: I do not know.'."
        + SUFFIX,
    },
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=60,
    ),
)

while True:
    i = input("You: ")
    send_message(agent.run(input=f'Human: "{i}"'))
