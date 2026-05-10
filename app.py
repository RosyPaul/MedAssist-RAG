from flask import Flask, render_template, jsonify, request
from src.helpers import download_embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GROQ_API_KEY"]=GROQ_API_KEY


embeddings = download_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # fast and free
    api_key=GROQ_API_KEY,
    temperature=0.5,
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- RAG Chain ----
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)