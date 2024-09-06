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

st.header("Testing")
