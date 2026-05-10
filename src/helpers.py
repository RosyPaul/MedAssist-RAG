from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Extracting text from pdf file
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] =[]

    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

def text_splitter(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts_chunk=text_splitter.split_documents(minimal_docs)
    return texts_chunk



def download_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embeddings