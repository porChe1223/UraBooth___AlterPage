# GA4DataGetter

Google Analytics からのデータを取得する関数  
https://ga4datagetter.azurewebsites.net/data

## 環境構築

python3 -m venv .venv  
F1  
Azure Functions: Download Remote Settings...

#### 注意

Python 3.11.11

## ローカルデバック

F5

### トラブルシューティング

#### 基本

. .venv/bin/activate  
pip install -r requirements.txt  
func host start

#### Module Not Found

rm -rf .venv  
python3 -m venv .venv
F5

#### ポートがすでに使われている場合

lsof -i  
kill -9 <localhost:9091 の PID>

## デプロイ

Azure  
RESOURCES  
Azure Subscription  
Function App  
対象の関数を右クリック  
Create Function App in Azure...

## その他

今回は GA4 のレポート結果をメトリクス順に指定していない。理由はディメンションとメトリクスを 1:1 対応でレポートを作成させているからだ。  
もしメトリクスを一気に複数選択し、レポート結果をメトリクス順に指定したい場合、コメントアウトの部分を戻せばできるようになる。

# CosmosDBDataGetter

Cosmos Database からのデータを取得する関数  
https://cosmosdbdatagetter.azurewebsites.net/data

## 環境構築

python3 -m venv .venv  
F1  
Azure Functions: Download Remote Settings...

#### 注意

Python 3.11.11

## ローカルデバック

F5

### トラブルシューティング

#### 基本

. .venv/bin/activate  
pip install -r requirements.txt  
func host start

#### Module Not Found

rm -rf .venv  
python3 -m venv .venv
F5

#### ポートがすでに使われている場合

lsof -i  
kill -9 <localhost:9091 の PID>

## デプロイ

Azure  
RESOURCES  
Azure Subscription  
Function App  
対象の関数を右クリック  
Create Function App in Azure...

# Dify

Dify で LLM ワークフロー作成  
https://udify.app/chat/TAeetpmo7e9l1pYx

## 環境構築

Dify を開く(https://cloud.dify.ai/apps)  
スタジオ  
アプリを作成する  
DSL ファイルをインポート  
ディレクトリ内の yml ファイルを選択

## yml ファイル保存

タイトルをクリック  
DSL をエクスポート

## デプロイ

公開する  
アプリを実行

# LLMDataAnalyzer

## 環境構築

- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

### WindowsPC の場合

- export FLASK_APP=app_analytics
- set FLASK_APP=app_analytics
- flask run

### MacOS の場合

#### 注意

- Python 3.11.11

# Slackbot

slackbot を作成して dify のチャットアプリでの応答を slack 上で表示

## 環境構築

.env ファイルを作成して環境変数を設定  
python3 -m venv venv  
pip install -r requirements.txt

## デバック

source venv/bin/activate
python3 connect_dify.py
