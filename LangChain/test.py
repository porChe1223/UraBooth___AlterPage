from azure.cosmos import CosmosClient
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
import json

# APIキーとエンドポイントの取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
DATABASE_NAME = "cosmosdbdatagetter"
CONTAINER_NAME = "my-container"

# OpenAI APIキーの設定
openai.api_key = OPENAI_API_KEY

# Cosmos DBクライアントの初期化
client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)

# OpenAI埋め込みモデルの初期化
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# データをベクトル化してCosmos DBに保存する関数
def vectorize_and_store_data():
    items = container.query_items(
        query="SELECT c.id, c.text FROM c",
        enable_cross_partition_query=True
    )

    for item in items:
        text = item["text"]
        embedding = embeddings_model.embed_query(text)

        # 埋め込みベクトルを保存
        container.upsert_item({
            "id": item["id"],
            "text": text,
            "embedding": embedding
        })

# ベクトル検索を実行する関数
def vector_search(query, top_k=5):
    # クエリのベクトル化
    query_vector = embeddings_model.embed_query(query)

    # ベクトル検索クエリ
    query_text = (
        "SELECT TOP @top_k c.id, c.text, VECTOR_DISTANCE(c.embedding, @query_vector) AS score "
        "FROM c "
        "ORDER BY score ASC"
    )

    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@query_vector", "value": query_vector}
    ]

    results = container.query_items(query=query_text, parameters=parameters, enable_cross_partition_query=True)

    return list(results)

if __name__ == "__main__":
    # データのベクトル化と保存
    print("Vectorizing and storing data...")
    vectorize_and_store_data()

    # クエリ入力
    query_input = "ブログの情報を取得してください"

    # ベクトル検索の実行
    print("Performing vector search...")
    search_results = vector_search(query_input)

    # 検索結果の表示
    print("Search results:")
    for result in search_results:
        print(json.dumps(result, indent=2, ensure_ascii=False))
