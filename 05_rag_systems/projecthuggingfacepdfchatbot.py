import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Local Embeddings
from langchain_ollama import ChatOllama # Local LLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA, LLMChain # Legacy support
from langchain_core.prompts import PromptTemplate
import pymupdf as fitz
from PIL import Image
import io
import pandas as pd

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Using Llama3 (Free/Local). Ensure Ollama is running!
MODEL_NAME = "llama3:latest" 

# Initialize Local LLM
llm = ChatOllama(model=MODEL_NAME, temperature=0.4)

# Define the QA prompt template
PROMPT = PromptTemplate(
    template="""Context: {context}
Question: {question}
Answer concisely based on the context. If unsure, say you don't know.""",
    input_variables=["context", "question"]
)

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted: text += extracted
    
    # Updated import path
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_text(text)

def create_embeddings_and_vectorstore(texts):
    """Uses CPU-based HuggingFace embeddings (Free)"""
    print("Generating local embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def expand_query(query: str, llm: ChatOllama) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="List 3 keywords related to: {query}. Output only keywords separated by commas."
    )
    # Using modern pipe syntax instead of LLMChain
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return f"{query} {response.content}"

def rag_pipeline(query, qa_chain, vectorstore):
    expanded_query = expand_query(query, llm)
    # Get relevant docs
    response = qa_chain.invoke({"query": query})
    return response['result'], expanded_query

def gradio_interface(pdf_file, query):
    texts = process_pdf(pdf_file.name)
    vectorstore = create_embeddings_and_vectorstore(texts)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    result, expanded = rag_pipeline(query, qa, vectorstore)
    return result, f"Expanded Query: {expanded}\nChunks: {len(texts)}"

# Launch Interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Question")],
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Log")],
    title="Local PDF QA (No API Key Needed)"
)

if __name__ == "__main__":
    iface.launch()