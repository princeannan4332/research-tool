import streamlit as st
import os
import pandas as pd
import PyPDF2
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv



st.title("Nutri Kid AI Research Tool 🧑‍🍼")
st.sidebar.title("News Article Or Research URLs")


urls = []
# Initialize session state if not already initialized
if 'input_count' not in st.session_state:
    st.session_state.input_count = 0  # Tracks how many input fields we have

# Function to add a new input field
def add_input():
    st.session_state.input_count += 1
    st.session_state.has_input=True

def remove_input():
    if  "input_count" in st.session_state:
            st.session_state.input_count = st.session_state.input_count-1

# Display the input fields based on how many clicks (input_count)
for i in range(st.session_state.input_count):
    inp=st.sidebar.text_input(f"Input {i+1}")
    urls.append(inp)


# Button to add new input fields
if st.sidebar.button("Add URL Input Field"):
    add_input()

# Button remove new input fields
if st.sidebar.button("Remove URL Input Field"):
    remove_input()



#Custorm docs uploaded by the user
custorm_docs=[]

# File uploader to accept file input from user
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file
    file_content = uploaded_file.read().decode("utf-8")
    
    doc=Document(page_content=file_content, metadata={"source":uploaded_file.name})
    custorm_docs.append(doc)


# File uploader to accept CSV file
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    doc=Document(page_content=df, metadata={"source":uploaded_file.name})
    custorm_docs.append(doc)



# File uploader to accept Excel file
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)
    
    doc=Document(page_content=df, metadata={"source":uploaded_file.name})
    custorm_docs.append(doc)



# File uploader to accept PDF file
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    
    # Extract text from each page
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    doc=Document(page_content=text, metadata={"source":uploaded_file.name})
    custorm_docs.append(doc)








process_url_clicked=st.sidebar.button("Process Info")

file_path = "faiss_store_openai"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)


if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    url_data = loader.load()

    data=url_data+custorm_docs
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore=PineconeVectorStore.from_documents(docs,embeddings,index_name="pinecone-v1")
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)







index_name='pinecone-v1'#name of the index we created in pinecone

query = main_placeholder.text_input("Question: ")
if query:
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)



