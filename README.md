# GA4DataGetter

Google Analytics からのデータを取得する関数  
https://ga4datagetter.azurewebsites.net/data

## 環境構築

python3 -m venv .venv  
F1  
Azure Functions: Download Remote Settings...

#### 注意

Python 3.11.10

## ローカルデバック

F5

### 失敗時

. .venv/bin/activate  
pip install -r requirements.txt  
func host start

### ポートがすでに使われている場合

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

Python 3.11.10

## ローカルデバック

F5

### 失敗時

. .venv/bin/activate  
pip install -r requirements.txt  
func host start

### ポートがすでに使われている場合

lsof -i  
kill -9 <localhost:9091 の PID>

## デプロイ

Azure  
RESOURCES  
Azure Subscription  
Function App  
対象の関数を右クリック  
Create Function App in Azure...

<!-- # LLMDataAnalyzer

## 手順

- python3 -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- export FLASK_APP=app_analytics
- set FLASK_APP=app_analytics
- flask run

#### 注意

- Python 3.11.11 -->
