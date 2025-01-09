import azure.functions as func
import logging
from azure.cosmos import CosmosClient, exceptions
import json
import os

COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
ITEM_ID = '2022-12-28 to 2025-1-1'

app = func.FunctionApp()

@app.function_name(name="GetDBInfo")
@app.route(route="data", auth_level=func.AuthLevel.ANONYMOUS)
def GetDBInfo(req: func.HttpRequest) -> func.HttpResponse:

    # Cosmos DB から情報を取得
    try:
        client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
        database_name = "my-database"
        database = client.get_database_client(database_name)

        # データベース内の全コンテナのリストを取得
        containers_list = database.list_containers()
        containers_names = [container['id'] for container in containers_list]

        all_data = {}

        # 各コンテナから指定したアイテムを取得
        for container_name in containers_names:
            container = database.get_container_client(container_name)
            try:
                item = container.read_item(item=ITEM_ID, partition_key=ITEM_ID)
                all_data[container_name] = item
            except exceptions.CosmosResourceNotFoundError:
                all_data[container_name] = None
        
        response_body = json.dumps(all_data, ensure_ascii=False).replace('\\n', '\n').encode('utf-8')

        return func.HttpResponse(
            body=response_body,
            status_code=200,
            mimetype="application/json",
            charset="utf-8"
        )

    except exceptions.CosmosHttpResponseError as e:
        logging.error(f'Cosmos DB error: {e}')
        return func.HttpResponse(f'Cosmos DB error: {e}', status_code=500)
    
    except Exception as e:
        logging.error(f'Error: {e}')
        return func.HttpResponse(f'Error: {e}', status_code=500)
