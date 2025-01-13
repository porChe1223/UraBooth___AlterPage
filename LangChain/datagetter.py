import requests
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
import os

vitamin = os.getenv("OPENAI_API_KEY")

url = "https://cosmosdbdatagetter.azurewebsites.net/data"

def custom_loader(url):  # 引数は単一のURL
    documents = []
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if "my-container" in data:
                documents.extend([item["data"] for item in data["my-container"] if "data" in item])
            else:
                print(f"Expected 'my-container' in the data, but not found: {data}")
        except requests.exceptions.JSONDecodeError as e:
            print(f"Error decoding JSON from {url}: {e}")
            print(f"Response content: {response.text}")
    else:
        print(f"Failed to fetch data from {url}, status code: {response.status_code}")
    return documents  # documentsを返す

documents = custom_loader(url)  # 単一のURLを渡す

if not documents:
    raise ValueError("No documents loaded from the provided URL")

embeddings = OpenAIEmbeddings(api_key=vitamin)
vectorstore = Chroma(embedding=embeddings).from_documents(documents)

# `documents`の表示を制限して数が多い場合も問題なく処理されるように
# print(f"Loaded {len(documents)} documents")  # 必要に応じてデバッグ用に表示


# データを一部表示して確認
# print(f"Loaded {len(documents)} documents")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=0,
    length_function=len,
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(api_key=vitamin),
    text_splitter=text_splitter,
).from_documents(documents)

# 質問の処理
#query = "記事のタイトルは？"
#answer = index.query(query)
#print(f"Answer: {answer}")
