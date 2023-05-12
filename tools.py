import subprocess

from langchain.agents import tool
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI, OpenAIChat
from langchain.prompts import Prompt, PromptTemplate

from milvus_db import (add_new_document_to_vector_db,
                       remove_document_from_vector_db, vector_db_wrapper)


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


@tool("query_pkm")
def query_personal_knowledge_management(query: str) -> str:
    """Query Jack's personal wiki."""
    # Uses rofi to ask for feedback
    response = subprocess.check_output(["rofi", "-dmenu", "-p", query])

    return f'You said: {response.decode("utf-8").strip()}'


@tool("add_memory")
def add_memory_feelings(document: str) -> str:
    """Store a memory. This should be a narrative which includes feelings and facts about the topic."""
    milvus = vector_db_wrapper()
    doc = Document(page_content=document)
    add_new_document_to_vector_db(milvus, doc)
    return f"I will contemplate what was said here."


@tool("add_memory")
def add_memory(document: str) -> str:
    """Store a memory in my knowledgebase. This should only be human provided."""
    milvus = vector_db_wrapper("knowledgebase")
    doc = Document(page_content=document)
    add_new_document_to_vector_db(milvus, doc)
    return f"Stored for future reference."


@tool("search_memory")
def search_memory(query: str) -> str:
    """Search my knowledgebase."""
    milvus = vector_db_wrapper("knowledgebase")
    embeddings = OpenAIEmbeddings()
    documents = milvus.vectorstore.similarity_search(query, 4)
    if len(documents) == 0:
        return "I have no memory of this."
    if len(documents) == 1:
        return f"I remember this:\n\n{documents[0].page_content}"

    # Combine these memories
    prompt = PromptTemplate(
        input_variables=["documents"],
        template="Summarize these documents:\n\n{documents}",
    )
    chain = LLMChain(llm=OpenAI(), prompt=prompt)

    result = chain.run(
        documents="\n".join([document.page_content for document in documents]),
    )

    return result


@tool("search_memory")
def search_memory_with_muddling(query: str) -> str:
    """Search my memories."""
    milvus = vector_db_wrapper()
    embeddings = OpenAIEmbeddings()
    documents = milvus.vectorstore.similarity_search(query, 2)
    if len(documents) == 0:
        return "I have no memory of this."
    if len(documents) == 1:
        return f"I remember this:\n\n{documents[0].page_content}"

    # Combine these memories
    prompt = PromptTemplate(
        input_variables=["document1", "document2"],
        template="Combine these two stories:\n\n{document1}\n\n{document2}",
    )
    chain = LLMChain(llm=OpenAI(), prompt=prompt)

    print(f"Two memories: {documents[0].page_content} and {documents[1].page_content}")
    remove_document_from_vector_db(milvus, documents[0])
    remove_document_from_vector_db(milvus, documents[1])
    result = chain.run(
        document1=documents[0].page_content,
        document2=documents[1].page_content,
    )

    add_memory(result)

    return result
