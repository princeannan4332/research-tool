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


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('api_key')
os.environ["PINECONE_API_KEY"]= os.getenv('pinecone_api_key')#get from app.pinecone.io


st.header("Testing")
