import azure.functions as func
import logging
from azure.cosmos import CosmosClient, exceptions
import json
import os

COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")

app = func.FunctionApp()

@app.function_name(name="GetDBInfo")
@app.route(route="data", auth_level=func.AuthLevel.ANONYMOUS)
def GetDBInfo(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Cosmos DB から情報を取得
    try:
        endpoint = COSMOS_DB_ENDPOINT
        key = COSMOS_DB_KEY
        client = CosmosClient(endpoint, key)
        database_name = "my-database"
        database = client.get_database_client(database_name)

        # データベース内の全コンテナのリストを取得
        containers_list = database.list_containers()
        containers_names = [container['id'] for container in containers_list]

        all_data = {}

        # 各コンテナからデータを取得
        for container_name in containers_names:
            container = database.get_container_client(container_name)
            
            # クエリの実行
            query = "SELECT * FROM c"
            items = list(container.query_items(query, enable_cross_partition_query=True))
            
            # データを辞書に保存
            all_data[container_name] = items

        # データをHTTPレスポンスとして返す
        return func.HttpResponse(json.dumps(all_data), status_code=200, mimetype="application/json")
    except exceptions.CosmosHttpResponseError as e:
        logging.error(f'Cosmos DB error: {e}')
        return func.HttpResponse(f'Cosmos DB error: {e}', status_code=500)
    except Exception as e:
        logging.error(f'Error: {e}')
        return func.HttpResponse(f'Error: {e}', status_code=500)