import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma 
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb

st.set_page_config(layout="wide")

checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    low_cpu_mem_usage=True
)

@st.cache_resource
def data_ingestion():
    documents = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())

    if not documents:
        print("No PDF files found in docs/. Nothing to ingest.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    client = chromadb.PersistentClient(path="db")
    db = Chroma.from_documents(
        texts,
        embeddings,
        client=client,
        collection_name="pdf-docs",
    )
    db = None

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings)
    retriever = db.as_retriever()
    
    # Create a simple prompt template
    template = """Use the following context to answer the question.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create a simple chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain, retriever

def process_answer(instruction):
    chain, retriever = qa_llm()
    
    # Get relevant documents
    docs = retriever.invoke(instruction)
    
    # Run the chain
    answer = chain.invoke(instruction)
    
    return answer

# Display conversation history using Streamlit chat UI
def display_conversation(history):
    for i in range(len(history["generated"])):
        with st.chat_message("user"):
            st.write(history["past"][i])
        with st.chat_message("assistant"):
            st.write(history["generated"][i])

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align:center; color: blue;'>Chat with Your PDF 🐦📄</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color: grey;'>Built by <a href='https://github.com/darshanbm26'>Darshan B M with ❤</a></h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color: red;'>Upload Your PDF Below👇</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_details = {
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "size": uploaded_file.size,
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1,col2 = st.columns((1,2))
        with col1:
            st.markdown("<h2 style='text-align:center;color:grey;'>PDF Details</h2>",unsafe_allow_html=True)
            st.write(file_details)
            st.markdown("<h2 style='text-align:center;color:grey;'>PDF Preview</h2>",unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner("Embeddings are in process..."):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h2 style='text-align:center; color: grey;'>Chat Here👇</h2>", unsafe_allow_html=True)    

            user_input = st.text_input("Enter your question", key="input", label_visibility="collapsed")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]
                
            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer(user_input)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(answer)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()