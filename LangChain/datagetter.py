from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from azure.cosmos import CosmosClient
import os
vitamin = os.getenv("OPENAI_API_KEY")
credential = os.getenv("COSMOS_DB_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=vitamin)


import langchain_community.document_loaders
import langchain.embeddings
import langchain.vectorstores
import json


documents_url = ["https://cosmosdbdatagetter.azurewebsites.net/data",]


loader = langchain_community.document_loaders.SeleniumURLLoader(urls=documents_url)  # 修正
documents = loader.load()
#print(f"出力ううううううううううう:{documents[:5]}") 
print(type(documents))  # 20文字に制限

client = CosmosClient(documents_url, credential)
my_container = client.get_container_client("my-container").get_container_client("documents")

embedding = langchain.embeddings.HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

def get_embedding(text, model):
   text = text.replace("\n", " ")
   res = openai.embeddings.create(input = [text], model=model).data[0].embedding
   return res

message = "ブログの情報を取得してください"
query_vector = get_embedding(message, model="text-embedding-ada-002")

for item in my_container.query_items(
   query="SELECT TOP 5 c.title, c.review, VectorDistance(c.contentVector,@embedding) AS SimilarityScore FROM c ORDER BY VectorDistance(c.contentVector,@embedding)",
   parameters=[
      {"name": "@embedding", "value": query_vector}
   ],
   enable_cross_partition_query=True):
   print(json.dumps(item, indent=True, ensure_ascii=False))


#vectorstore = langchain.vectorstores.Chroma.from_documents(
    #documents=documents,
    #embedding=embedding
#)