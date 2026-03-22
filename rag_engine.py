import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
load_dotenv()


def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    print(f"Loaded {len(documents)} pages from: {file_path}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, save_path="faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"Vector store saved to '{save_path}/' folder")
    return vector_store


def load_vector_store(save_path="faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Loaded existing vector store from '{save_path}/'")
    return vector_store


def build_qa_chain(vector_store):
    # Step 1 — retriever finds top 4 matching chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Step 2 — prompt tells GPT exactly what to do
    prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions
based strictly on the document provided below.

If the answer is not found in the document, respond with:
"I don't know based on the provided document."

Do NOT make up information. Do NOT use outside knowledge.

Document context:
{context}

User question: {question}

Your answer:""")

    # Step 3 — the LLM that generates the answer
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


    # Step 4 — helper function to join all chunks into one string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Step 5 — chain: question → retrieve → format → prompt → LLM → answer
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("QA Chain is ready!")
    return chain


def ingest_and_build(file_path: str):
    docs         = load_document(file_path)
    chunks       = split_documents(docs)
    vector_store = create_vector_store(chunks)
    chain        = build_qa_chain(vector_store)
    return chain