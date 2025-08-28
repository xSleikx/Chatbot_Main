import ollama
import os 
from langchain_community.document_loaders import PyMuPDFLoader


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Function to load, split, and retrieve documents
def load_and_retrieve_docs(pdf_file, question, chunk_size_slider, overlap_slider):
    loader = PyMuPDFLoader(pdf_file.name)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_slider, chunk_overlap=overlap_slider)
    docs = text_splitter.split_documents(pages)

    # Create Ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retrieve relevant documents based on the question
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)


    return retrieved_docs
