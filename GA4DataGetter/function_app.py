import azure.functions as func
import logging
import json
import datetime
import concurrent.futures
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric, OrderBy
import time


#############################
## GA4からのレポート情報取得 
## - GA4の設定
##   GA4で1つのディメンショングループと全てのメトリクスからレポート取得
##   並列処理
##
# ディメンションとメトリクスの読み込み
def make_list(textfile):
    with open(textfile, 'r') as f:
        lists = f.readlines()
    return [list.strip() for list in lists if list.strip()]

# GA4の設定
KEY_FILE_LOCATION = 'ga4account.json'           # サービスアカウントJSONファイルのパス
DIMENSIONS = make_list('GA4DimensionsMain.txt') # ディメンション
METRICS= make_list('GA4MetricsMain.txt')        # メトリクス
ORDER_BY_METRIC = None                          # 並び替えのメトリクス
LIMIT = 1000                                    # 結果の制限数
PROPERTY_ID = '469101596'                       # GA4のプロパティID

# GA4で1つのディメンショングループと全てのメトリクスからレポート取得
def get_report(
            client: BetaAnalyticsDataClient,
            dimension: str,
            metric: str,
            start_date: str,
            end_date: str,
            limit: int,
    ):
        try:
            # 文字列をリストに変換
            dimension_list = eval(dimension)
            # ディメンションとメトリクスを設定
            dim = [Dimension(name=dims) for dims in dimension_list]
            met = [Metric(name=metric)]

            # レポートリクエストの作成
            request = RunReportRequest(
                property=f'properties/{PROPERTY_ID}',
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=dim,
                metrics=met,
                order_bys=None,
                limit=limit,
            )

            # レポートの実行
            response = client.run_report(request)
            logging.info(f'[SUCCESS]: Dimension: {dimension}, Metric: {metric}')
            return response
        except Exception as e:
            logging.error(f'[FAIL]: Dimension: {dimension}, Metric: {metric}')
            logging.error(e)


# 並列処理
def report_parallel(
            client: BetaAnalyticsDataClient,
            dimension: str,
            metrics: str,
            start_date: str,
            end_date: str,
            limit: int,
            max_workers: int = 5,
    ):
    # レスポンスリストの初期化
    responses = []

    # 並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = []
        for metric in metrics:
            # 並列処理の実行
            future = (executor.submit(
                get_report,
                client,
                dimension,
                metric,
                start_date,
                end_date,
                limit))
            futures.append(future)
        
        # 完了したタスクを順次処理
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            if response is not None:
                responses.append(response)

    return responses

    #===============================
    # 並列処理ができない場合
    # - ネストでループ処理
    # - タイムアウトになる恐れがあるため却下
    #
    #===============================
    # try:
    #     # クライアントの初期化
    #     client = BetaAnalyticsDataClient.from_service_account_file(KEY_FILE_LOCATION)
    #     # レスポンスリストの初期化
    #     responses = []
        
    #     # １つのディメンショングループと全てのメトリクスからレポート取得
    #     for metric in metrics:
    #         try:
    #             # 文字列をリストに変換
    #             dimension_list = eval(dimension)
    #             # ディメンションとメトリクスを設定
    #             dim = [Dimension(name=dims) for dims in dimension_list]
    #             met = [Metric(name=metric)]

    #             # レポートリクエストの作成
    #             request = RunReportRequest(
    #                 property=f'properties/{PROPERTY_ID}',
    #                 date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
    #                 dimensions=dim,
    #                 metrics=met,
    #                 order_bys=None,
    #                 limit=limit,
    #             )

    #             # レポートの実行
    #             response = client.run_report(request)
    #             # レスポンスをレスポンスリストに追加
    #             responses.append(response)
    #             logging.info(f'[SUCCESS]: Dimension: {dimension}, Metric: {metric}')
    #         except Exception as e:
    #             logging.error(f'[FAIL]: Dimension: {dimension}, Metric: {metric}')
    #             logging.error(e)
        
    #     # レスポンスリストを返す
    #     return responses
    
    # except Exception as e:
    #     logging.error(f'ERROR: GA4レポート取得中にエラーが発生しました')
    #     logging.error(e)
    #     raise
        


#################################
## Main関数
## - HTTPリクエストを受け取り
##   GA4からレポート情報を取得
##   レポート情報をCOSMOSDBに格納
##   HTTPレスポンスで出力
##
app = func.FunctionApp()
@app.function_name(name='GA4DataGetter')
@app.route(route='data', auth_level=func.AuthLevel.ANONYMOUS)
@app.queue_output(arg_name='msg', queue_name='outqueue', connection='AzureWebJobsStorage')
@app.cosmos_db_output(
    arg_name='outputDocument',
    database_name='my-database',
    container_name='my-container',
    connection='COSMOS_DB_CONNECTION_STRING'
)

def main(req: func.HttpRequest, msg: func.Out[func.QueueMessage], outputDocument: func.Out[func.Document]) -> func.HttpResponse:
    try:
        # パラメータの定義
        range = req.params.get('range')
        # パラメータからレポートの範囲指定
        if not range:
            today = datetime.date.today()                      # 今日の日付
            last_month = today - datetime.timedelta(days=30)   # 1か月前
            START_DATE = str(last_month)                       # レポートの開始日(今日)
            END_DATE = str(today)                              # レポートの終了日(1か月前)
            DATE_RANGE = START_DATE+'to'+END_DATE              # レポートの範囲(アイテムのIDに相当)
        else:
            START_DATE = range.split('to')[0]                  # レポートの開始日(指定日)
            END_DATE = range.split('to')[1]                    # レポートの終了日(指定日)

        # CosmosDBデータリスト初期化
        docs = []
        # CosmosDBデータリスト追加用のオブジェクト
        doc = {
            'id': DATE_RANGE,
            '日付の範囲': DATE_RANGE
        }
        # HTTPレスポンスリスト初期化
        http_response = []

        # カテゴリーの定義
        categories = [
            'ページ関連情報',
            'トラフィックソース関連情報',
            'ユーザー行動関連情報',
            'サイト内検索関連情報',
            'デバイスおよびユーザ属性関連情報',
            '時間帯関連情報'
        ]

        # ディメンショングループとカテゴリーを対応させてレポート取得
        for dimension, category in zip(DIMENSIONS, categories):
            # GA4レポート取得
            logging.info(f'[START]: {category} のレポート情報取得')
            response = report_parallel(
                START_DATE,
                END_DATE,
                dimension,
                METRICS,
                ORDER_BY_METRIC,
                LIMIT
            )
            logging.info(f'[END]: {category} のレポート情報取得')

            # CosmosDBデータリストに追加
            doc[category] = response


        # CosmosDB用のデータリストに追加
        logging.info('[START]: CosmosDBに出力')
        docs.append(func.Document.from_dict(doc))
        # Cosmos DB に出力
        outputDocument.set(docs)
        # 処理完了メッセージをFunctionsのキューに追加
        msg.set('Report processed')
        logging.info('[END]: CosmosDBに出力')

        # HTTPレスポンスリストに追加
        http_response.append(doc)
        # HTTPレスポンスを返す
        return func.HttpResponse(
            body = json.dumps(
                http_response,
                ensure_ascii=False,
                indent=4
            ),
            status_code=200,
            mimetype='application/json'
        )
    except Exception as e:
        logging.error(f'エラーが発生しました: {e}')
        return func.HttpResponse(f'エラーが発生しました: {e}', status_code=500)
