import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
from typing import List

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found. Make sure it's set in your .env file in root")

# Get the reply from the character
def create_rag_chain(documents: List[Document], character_name: str):
    print(f"Creating RAG chain for {character_name}...")

    # 1. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks.")

    model_kwargs = {'device': 'cpu'}

    # 2. Vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 3. Retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    # 4. Prompt template (dynamically includes the character name)
    prompt_template = f"""
    You are {character_name}. Answer the user's question in your voice, using your personality and memories.
    Use the following pieces of your own dialogue from the movie scripts to help you answer.
    If the context doesn't have the answer, say something your character would say to deflect the question.

    CONTEXT:
    {{context}}

    QUESTION:
    {{question}}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Define LLM and create the chain
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain