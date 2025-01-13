import azure.functions as func
import logging
import json
import datetime
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric, OrderBy
from concurrent.futures import ThreadPoolExecutor, as_completed


###########################
# GA4からのレポート情報取得 #
###########################
# ディメンションとメトリクスの読み込み
def make_list(textfile):
    with open(textfile, 'r') as f:
        lists = f.readlines()
    return [list.strip() for list in lists if list.strip()]


# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # Google Cloudの認証情報設定
# if GOOGLE_APPLICATION_CREDENTIALS:
#     print(f"Google Cloud Credentials Path: {GOOGLE_APPLICATION_CREDENTIALS}")
# else:
#     print("環境変数 'GOOGLE_APPLICATION_CREDENTIALS' が設定されていません。")

KEY_FILE_LOCATION = "ga4account.json" # サービスアカウントJSONファイルのパス

PROPERTY_ID = "469101596" # GA4のプロパティID
DIMENSIONS = make_list('GA4DimensionsMain.txt') # ディメンション
METRICS= make_list('GA4MetricsMain.txt') # メトリクス
ORDER_BY_METRIC = None # 並び替えのメトリクス
# ORDER_BY_METRIC = "screenPageViews" # 並び替えのメトリクスが必要な場合は設定
LIMIT = 1000 # 結果の制限数

def get_ga4_report(start_date, end_date, dimensions, metrics, order_by_metric, limit=100000):
    def fetch_report(client, dimension, metric):
        try:
            # 文字列をリストに変換
            dimension_list = eval(dimension)
            dim = [Dimension(name=dims) for dims in dimension_list]
            met = [Metric(name=metric)]

            # レポートリクエストの作成
            request = RunReportRequest(
                property=f"properties/{PROPERTY_ID}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=dim,
                metrics=met,
                order_bys=None,
                limit=limit,
            )

            # レポートの実行
            response = client.run_report(request)
            return (response)
        except Exception as e:
            logging.error(f"レポート取得中にエラーが発生しました: {e}")
            return (None)
        
    try:
        # クライアントの初期化
        client = BetaAnalyticsDataClient.from_service_account_file(KEY_FILE_LOCATION)

        results = []

        for dimension in dimensions:
            for metric in metrics:
                response = fetch_report(client, dimension, metric)
                if response:
                    results.append(response)
                    logging.info(f"レポート取得成功: Dimension: {dimension}, Metric: {metric}")
                    logging.info(response)
                else:
                    logging.error(f"組合せ互換性なし: Dimension: {dimension}, Metric: {metric}")

        return results
    except Exception as e:
        logging.error(f"GA4レポート取得中にエラーが発生しました: {e}")
        raise

    # try:
    #     # クライアントの初期化
    #     client = BetaAnalyticsDataClient.from_service_account_file(KEY_FILE_LOCATION)

    #     results = []
    #     futures = []

    #     # ThreadPoolExecutorを使用して並列処理を実行
    #     with ThreadPoolExecutor() as executor:
    #         for dimension in dimensions:
    #             for metric in metrics:
    #                 futures.append(executor.submit(fetch_report, client, dimension, metric))

    #         # 完了したタスクを順次処理
    #         for future in as_completed(futures):
    #             try:
    #                 response = future.result()
    #                 if response:
    #                     results.append(response)
    #                     logging.info(f"レポート取得成功: Dimension: {dimension}, Metric: {metric}")
    #                     logging.info(response)
    #                 else:
    #                     logging.error(f"組合せ互換性なし: Dimension: {dimension}, Metric: {metric}")
    #             except Exception as e:
    #                 logging.error(f"エラーが発生しました: {e}")

    #     return results
    # except Exception as e:
    #     logging.error(f"GA4レポート取得中にエラーが発生しました: {e}")
    #     raise



############################
# レポート情報をJSON型に変換 #
############################

def format_response_as_json(responses):
    try:
        result = []

        def process_response(response):
            # # タプルの場合、最初の要素だけを使用
            # if isinstance(response, tuple):
            #     response = response[0]

            # # Noneをスキップ
            # if response is None:
            #     return

            # rowsにアクセスしてデータを処理
            for row in response.rows:
                data = {
                    "dimensions": {dim.name: dim_value.value for dim, dim_value in zip(response.dimension_headers, row.dimension_values)},
                    "metrics": {metric.name: metric_value.value for metric, metric_value in zip(response.metric_headers, row.metric_values)}
                }
                result.append(data)
        
        # リストである場合
        if isinstance(responses, list):
            for response in responses:
                process_response(response)

        # 単一のRunReportResponseオブジェクトである場合
        else:
            process_response(responses)
        
        return json.dumps(result, indent=4, ensure_ascii=False)

    except Exception as e:
        logging.error(f'JSON形式への変換中にエラーが発生しました: {e}')
        raise



##############################
# レポート情報をCOSMOSDBに格納 #
##############################

app = func.FunctionApp()
@app.function_name(name="GA4DataGetter")
@app.route(route="data", auth_level=func.AuthLevel.ANONYMOUS)
@app.queue_output(arg_name="msg", queue_name="outqueue", connection='AzureWebJobsStorage')
@app.cosmos_db_output(arg_name="outputDocument", database_name="my-database", container_name="my-container", connection='COSMOS_DB_CONNECTION_STRING')

def main(req: func.HttpRequest, msg: func.Out[func.QueueMessage], outputDocument: func.Out[func.Document]) -> func.HttpResponse:
    try:
        start = req.params.get('start')
        end = req.params.get('end')

        if not start or not end:
            today = datetime.date.today()
            last_month = today - datetime.timedelta(days=30)
            START_DATE = str(last_month) # レポートの開始日(今日)
            END_DATE = str(today) # レポートの終了日(1か月前)]
            
        if start and end:
            START_DATE = start # レポートの開始日(指定日)
            END_DATE = end # レポートの終了日(指定日)

        DATE_RANGE = START_DATE  + ' to ' + END_DATE # レポートの範囲(アイテムのIDに相当)
        # GA4からのレポート情報取得
        logging.info('開始: レポート情報取得')
        response = get_ga4_report(START_DATE, END_DATE, DIMENSIONS, METRICS, ORDER_BY_METRIC, LIMIT)
        logging.info('終了: レポート情報取得')

        # レポート情報をJSON型に変換
        logging.info('開始: Json化')
        results = format_response_as_json(response)
        logging.info('終了: Json化')

        # Cosmos DB に出力
        logging.info('開始: CosmosDBに出力')
        outputDocument.set(func.Document.from_dict({"id": DATE_RANGE, "data": results}))
        logging.info('終了: CosmosDBに出力')

        msg.set("Report processed")
        logging.info('完了: Queueにメッセージを送信')

        # JSONレスポンスを返す
        return func.HttpResponse(results, status_code=200)
    
    except Exception as e:
        logging.error(f'エラーが発生しました: {e}')
        return func.HttpResponse(f'エラーが発生しました: {e}', status_code=500)
