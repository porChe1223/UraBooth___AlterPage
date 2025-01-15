
import os

#TODO: APIキーの登録が必要
openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import openai

#def data_get(llm, prompt):

llm = ChatOpenAI(model_name="gpt-4o-mini",
                temperature=0,
                openai_api_key=openai_api_key
                )

urls = [
    "https://zenn.dev/umi_mori/articles/what-is-gpt-4",
    "https://zenn.dev/umi_mori/articles/chatgpt-api-python",
    "https://zenn.dev/umi_mori/articles/chatgpt-google-chrome-plugins",
]

urls2 = "https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報"


loader = UnstructuredURLLoader(urls=urls2)
print(loader.load())

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 300,
    chunk_overlap = 0,
    length_function = len,
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])

print("あああああああああああああああああ",index)
#return index



query = "7番目に紹介しているChatGPT便利プラグインは？"
answer = index.query(query,llm)
print(answer)
