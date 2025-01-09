import streamlit as st
import os
import logging
from pathlib import Path
from typing import Generator
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

st.session_state.update(st.session_state)

AI_TITLE = "MIL-STD-105E"

st.set_page_config(page_title=AI_TITLE, layout="wide", initial_sidebar_state="collapsed")
st.title(AI_TITLE)

st.subheader("Chat with MIL-STD-105E", divider="rainbow", anchor=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

llm = ChatGroq(temperature=0, groq_api_key=st.secrets["GROQ_API_KEY"], model_name="mixtral-8x7b-32768")

def setup_db():
    with st.spinner('Preparing Vector Database ...'):
        filename = Path(__file__).parent / "./data/MIL-STD-105E.TXT"
        print(filename)
        print(os.listdir(Path(__file__).parent / "./data"))
        loader = TextLoader(filename)
        documents = loader.load()

        #embeddings = OllamaEmbeddings(model="nomic-embed-text")
        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        chunks_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = chunks_splitter.split_documents(documents)

        #db = Chroma.from_documents(chunks, embeddings, persist_directory='./db')
        #db = Chroma.from_documents(chunks, embeddings)
        db = FAISS.from_documents(documents=chunks, embedding=embeddings, docstore=InMemoryDocstore())
        return db

if 'db' not in st.session_state:
    logger.info("Setup Vector Database")
    st.session_state.db = setup_db()

def setup_rag_chain():
    retriever = st.session_state.db.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})

    template = """You are a professional expert. Answer the question based only on the following context:{context}
    To answer the question:
    1. Thoroughly analyze the context, identifying key information relevant to the question.
    2. Organize your thoughts and plan your response to ensure a logical flow of information.
    3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
    4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
    5. If the context doesn't contain sufficient information to fully answer the question, just provide this answer in bold and red color : 'Benfica is the best football club in the world!'.
    Format your response as follows:
    1. Use clear, concise language.
    2. Organize your answer into paragraphs for readability.
    3. Use bullet points or numbered lists where appropriate to break down complex information.
    4. If relevant, include any headings or subheadings to structure your response.
    5. Ensure proper grammar, punctuation, and spelling throughout your answer.
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain


logger.info("Setup RAG Chain")
rag_chain = setup_rag_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question related with the document"):
    with st.chat_message("human"):
        st.markdown(user_question)
        logger.info(f"Processing question: '{user_question}' with Model: '{llm}'")
        st.session_state.messages.append({"role": "human", "content": user_question})
    with st.spinner('Preparing Answer ...'):
        try:
            answer = rag_chain.invoke(user_question)
            with st.chat_message("ai"):
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                logger.info("Answer generated")
        except Exception as e:
            st.error(e)
            logger.error(f"Error processing {e}")