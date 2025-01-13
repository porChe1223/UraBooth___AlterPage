import requests
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
import os

# OpenAI APIキーの取得
vitamin = os.getenv("OPENAI_API_KEY")

# RequestsWrapperを用いてHTTPリクエストを行うカスタムツールの定義
@tool("fetch_http_data", return_direct=True)
def fetch_http_data(url: str, method: str = "GET", headers: dict = None, params: dict = None) -> dict:
    """
    Fetch data from the given URL using HTTP request with the specified method and parameters.
    """
    if headers is None:
        headers = {}
    if params is None:
        params = {}

    try:
        if method.lower() == "get":
            response = requests.get(url, headers=headers, params=params)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if response.status_code == 200:
            return response.json()  # JSON形式で返却
        else:
            raise ValueError(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    
    except Exception as e:
        return {"error": str(e)}

# OpenAI Chat LLMの設定
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=vitamin)

# Toolの設定
tools = [
    Tool(
        name="Fetch HTTP Data",
        func=fetch_http_data,
        description="Use this tool to fetch data from a URL via HTTP request."
    )
]

# エージェントの初期化
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# リクエストURLと設定
url = "https://cosmosdbdatagetter.azurewebsites.net/data"
headers = {}  # 必要な場合ヘッダーを追加
params = {}  # 必要な場合クエリパラメータを追加

# エージェントによる実行
response = agent.run(input=f"Fetch the data from {url} using GET method and process the result.")
print(response)

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
vectorstore = Chroma.from_documents(documents, embedding=embeddings)

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
# query = "記事のタイトルは？"
# answer = index.query(query)
# print(f"Answer: {answer}")
