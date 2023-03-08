from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI, OpenAIChat
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

llm = OpenAI(temperature=0)

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What would be a good company name that makes {product}? Only puns allowed.",
# )
#
# chain = LLMChain(llm=llm, prompt=prompt)
#
# print(chain.run(product="demotivational socks"))
memory = ConversationBufferMemory(memory_key="chat_history")

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    verbose=True,
    memory=memory,
)

while True:
    i = input("You: ")
    print(agent.run(input=i))
