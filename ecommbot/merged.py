from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

# Initialize OpenAI Embeddings
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Data ingestion and vector store initialization
from langchain.schema import Document

def ingestdata(status):
    vstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name="flipkartreviewdata",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    if status is None:
        # Example documents wrapped in Document objects
        docs = [
            Document(page_content="Affordable Bluetooth earbuds with great sound quality.", metadata={"category": "electronics"}),
            Document(page_content="High-quality bass headphones under budget.", metadata={"category": "electronics"}),
        ]
        inserted_ids = vstore.add_documents(docs)
        return vstore, inserted_ids
    else:
        return vstore

# Generation logic for the bot
def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommerce bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    llm = ChatOpenAI()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == "__main__":
    vstore, inserted_ids = ingestdata(None)
    print(f"\nInserted {len(inserted_ids)} documents.")
    results = vstore.similarity_search("low budget bass headphones")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")

    vstore = ingestdata("done")
    chain = generation(vstore)
    print(chain.invoke("Can you tell me the best Bluetooth earbuds?"))
