## Setup

### Install dependencies

```
poetry shell
poetry install

# Optionally install extras: load-raw-data, google-search

# Load raw data installs `unstructured`; Unstructured is a library used for parsing all sorts of files into text.
poetry install --extras load-raw-data

# Load raw data installs a search API wrapper
poetry install --extras google-search
```

### Update `.env` file.

```
cp .env.example .env
```

You will need an `OPENAI_API_KEY` at the very least. Visit https://platform.openai.com/account/api-keys to
create a new secret key.

## Examples

```python

# All examples with a __main__ entrypoint should call load_dotenv
from dotenv import load_dotenv

load_dotenv()

# Define your chain or agent here
# Example, here's a simple chatbot using ChatGPT's API

# Imports the Chain class
# A **Chain** is a combination of LLMs and Prompts.
from langchain.chains import ConversationChain
# Imports the memory
# **Memory** can be used to inject extra context into a prompt. This is useful for Chatbots.
from langchain.chains.conversation.memory import ConversationBufferMemory
# Imports the OpenAIChat LLM
from langchain.llms import OpenAIChat

# Define the LLMs
# Here we're using the OpenAIChat LLM
llm = OpenAIChat()

# Define the memory
# Here we're using the ConversationBufferMemory
memory = ConversationBufferMemory()

# Define the chain
# Here we're using the ConversationChain
chain = ConversationChain(llm=llm, memory=memory, verbose=True)

if __name__ == "__main__":
    while True:
        i = input("Human: ")
        print(chain(i)["response"])
```
