import azure.functions as func
import logging
from azure.cosmos import CosmosClient, exceptions
import json
import os

COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DB_NAME = "my-database"
CONTAINER_NAME = "my-container"

app = func.FunctionApp()

@app.function_name(name="CosmosDBDataGetter")
@app.route(route="data", auth_level=func.AuthLevel.ANONYMOUS)
def GetDBInfo(req: func.HttpRequest) -> func.HttpResponse:
    # 日付範囲をリクエストパラメータから取得
    data_range = req.params.get('data_range')

    # Cosmos DB から情報を取得
    try:
        # DB情報
        client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
        database = client.get_database_client(COSMOS_DB_NAME)
        container = database.get_container_client(CONTAINER_NAME)
        # 出力用変数
        all_data = {}

        if not data_range:
            # コンテナから全てのアイテムを取得
            items = list(container.read_all_items())
            for item in items:
                all_data[CONTAINER_NAME] = items

        if data_range:
            # コンテナから指定したアイテムを取得
            ITEM_ID = data_range
            try:
                item = container.read_item(item=ITEM_ID, partition_key=ITEM_ID)
                all_data[CONTAINER_NAME] = item
            except exceptions.CosmosResourceNotFoundError:
                all_data[CONTAINER_NAME] = 'Item Not Found'
        
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
