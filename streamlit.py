import os
import pandas as pd
import PyPDF2
import streamlit as st
import time
from langchain import OpenAI



load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('api_key')
os.environ["PINECONE_API_KEY"]= os.getenv('pinecone_api_key')#get from app.pinecone.io


st.header("Testing")
