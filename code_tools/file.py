"""
python code_tools/file.py when-is "How does the ask_question_about_project agent work?" /home/jack/Code/github.com/jack-michaud/langchain-playground/

> The ask_question_about_project agent uses the search_for_references tool to find files in the project that contain the given identifier using ripgrep. 
> It then uses the answer_question_about_code_in_file agent to answer questions about the code in the file.

python code_tools/file.py when-is "How does answer_question_about_code_in_file work?" /home/jack/Code/github.com/jack-michaud/langchain-playground/

> answer_question_about_code_in_file takes a filename and a question as input, loads the file, and summarizes the code to answer the question 
> if the file is small enough.

When a file is too big, it chunks it out into smaller files and uses the VectorDBQA agent to answer the question.
The VectorDBQA uses embedding search to find the most similar code to the question and then uses a LLM to answer the question.


"""
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain.agents.mrkl.prompt import SUFFIX
from langchain.chains import LLMChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAIChat
from langchain.prompts import PromptTemplate

load_dotenv()

from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def summarize_code(code: str, question: Optional[str] = None):
    llm = OpenAIChat()
    if question is None:
        prompt = PromptTemplate(
            input_variables=["code"],
            template="Please document the following code: {code}",
        )
    else:
        prompt = PromptTemplate(
            input_variables=["code", "question"],
            template="Please answer this question about the following code: {question}\n\n{code}",
        )
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain({"code": code, "question": question})


@tool("answer_question_about_code_in_file")
def answer_question_about_code_in_file(filename_and_question: str):
    """
    Summarize the code in the given file and answer the given question. The format of action input should be filename,question. Filename should be a full path.
    """
    filename, question = filename_and_question.split(",")
    # strip newlines and quotes from filename and question
    filename = filename.strip()
    filename = filename.strip('"')
    question = question.strip()
    question = question.strip('"')

    loader = TextLoader(filename)
    try:
        documents = loader.load()
    except FileNotFoundError:
        return "File not found. Please provide a full path to the file."
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    if len(texts) < 4:
        return summarize_code(
            "".join([d.page_content for d in texts]), question=question
        )["text"]

    embeddings = OpenAIEmbeddings()
    file = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
    return VectorStoreIndexWrapper(vectorstore=file).query(question)
    # return summarize_code(code, question=question)["text"]


def ask_question_about_project(action: str, project_path: Path):
    @tool("search_for_references")
    def search_for_references(identifier: str):
        """
        Find files in the project that contain the given identifier using ripgrep. Identifier must be a single word.
        """
        import subprocess

        # strip newlines and quotes from identifier
        identifier = identifier.strip()
        identifier = identifier.strip('"')
        identifier = identifier.strip("'")

        llm = OpenAIChat()
        try:
            matches = subprocess.check_output(
                [
                    "rg",
                    "-c",
                    identifier,
                    project_path.absolute().as_posix(),
                ]
            ).decode()
        except CalledProcessError:
            matches = (
                "Command failed. Did you provide only one word for the action input?"
            )

        # prompt = PromptTemplate(
        #     input_variables=["identifier", "matches"],
        #     template="Where else is {identifier} used given these matches in my project? \n\n{matches}",
        # )
        # chain = LLMChain(llm=llm, prompt=prompt)
        # return chain({"identifier": identifier, "matches": matches})
        return matches

    agent = initialize_agent(
        [
            search_for_references,
            answer_question_about_code_in_file,
        ],
        llm=OpenAIChat(),
        verbose=True,
        agent="zero-shot-react-description",
        max_iterations=10,
        agent_kwargs={
            "suffix": "A concrete example:\n\nAction: search_for_references\nAction Input: identifier\n\n"
            + SUFFIX
        },
    )
    return agent(
        f"{action} Feel free to ignore tests and other files that are not relevant."
    )["output"]


if __name__ == "__main__":
    import argparse

    root_parser = argparse.ArgumentParser()
    subparsers = root_parser.add_subparsers()

    # Add file question subparser
    file_question_parser = subparsers.add_parser(
        "file-question",
    )
    file_question_parser.add_argument("filename", type=Path)
    file_question_parser.add_argument("--question", type=str, default=None)

    # Add the "where else is this used" subparser
    where_else_used_parser = subparsers.add_parser(
        "when-is",
    )
    # where_else_used_parser.add_argument("identifier", type=str)
    where_else_used_parser.add_argument("action", type=str, default=None)
    where_else_used_parser.add_argument("project_path", type=Path)

    args = root_parser.parse_args()

    # If using the file-question subparser, use summarize_code_in_file to summarize the code in the file given file and ask the given question
    if hasattr(args, "filename"):
        print(
            answer_question_about_code_in_file(args.filename, question=args.question)[
                "text"
            ]
        )

    # If using the "where else is this used" subparser, use ask_about_references_to_class_or_function to ask about the given identifier in the given project
    if hasattr(args, "action"):
        for _ in range(3):
            try:
                print(ask_question_about_project(args.action, args.project_path))
                break
            except Exception as e:
                print("Error: ", e)
                print("Trying again...")
                continue
