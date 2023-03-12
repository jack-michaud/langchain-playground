from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Milvus, VectorStore

text_field = "document_text"


def get_vector_db(collection_name: str = "default") -> VectorStore:
    return Milvus(
        collection_name=collection_name,
        text_field=text_field,
        embedding_function=OpenAIEmbeddings(),
        connection_args={},
    )


def vector_db_wrapper(
    vectorstore: VectorStore | None = None,
) -> VectorStoreIndexWrapper:
    """
    # Example

    >>> from milvus_db import vector_db_wrapper
    >>> vectordb = vector_db_wrapper()
    >>> vectordb.query("What is the airspeed of a fully laden swallow?")

    """
    if vectorstore is None:
        vectorstore = get_vector_db()
    return VectorStoreIndexWrapper(vectorstore=vectorstore)


def add_new_document_to_vector_db(
    vector_db: VectorStoreIndexWrapper,
    document: Document,
):
    vector_db.vectorstore.add_documents([document])
