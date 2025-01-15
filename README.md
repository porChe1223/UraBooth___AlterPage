# GA4DataGetter

Google Analytics からのデータを取得する関数  
https://ga4datagetter.azurewebsites.net/data

## 使い方

ほしいデータの範囲を指定したいときは、URL の末尾に  
`?range=AAAA-BB-CCtoXXXX-YY-ZZ`
と追加してください  
例:  
`?range=2024-12-01to2024-12-31`

データの範囲指定がない場合は、今日から 1 月前までのレポートが取得されます

以下のようなデータが取得されます  
[  
 {  
 "id": "2024-12-16to2025-01-15",  
 "日付の範囲": "2024-12-16to2025-01-15",  
 "ページ関連情報": [],  
 "トラフィックソース関連情報": [],  
 "ユーザー行動関連情報": [],  
 "サイト内検索関連情報": [],  
 "デバイスおよびユーザ属性関連情報": [],  
 "時間帯関連情報": []  
 }  
]

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
Deploy to Function App...

# CosmosDBDataGetter

Cosmos Database からのデータを取得する関数  
https://cosmosdbdatagetter.azurewebsites.net/data

## 使い方

### 日付の範囲

日付の範囲を指定したいときは、URL の末尾に  
`?range=AAAA-BB-CCtoXXXX-YY-ZZ`
と追加してください  
例:  
`?range=2024-12-1to2024-12-31`

データの範囲指定がない場合は、今日から 1 月前までのレポートが取得されます

### ディメンションのグループ

グループを指定したいときは、URL の末尾に
`?group=<グループ名>`
と追加してください
例:  
`?group=ページ関連情報`
グループ名

- ページ関連情報
- トラフィックソース関連情報
- ユーザー行動関連情報
- サイト内検索関連情報
- デバイスおよびユーザ属性関連情報
- 時間帯関連情報

### 両方

両方指定する場合は後者に&をつけてください
例
`?range=2024-12-1to2024-12-31&group=ページ関連情報`

以下のようなデータが取得されます  
[  
 {  
 　"id": "2024-12-16to2025-01-15",  
　 "日付の範囲": "2024-12-16to2025-01-15",  
 　"ページ関連情報": [],  
　 "トラフィックソース関連情報": [],  
　 "ユーザー行動関連情報": [],  
　 "サイト内検索関連情報": [],  
 　"デバイスおよびユーザ属性関連情報": [],  
　 "時間帯関連情報": []  
 }  
]

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
Deploy to Function App...

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

# LangChain

## 環境構築

- requirements.txt を作成し、以下を書き加える。openai 以外の AI を使う場合は随時そのライブラリをインストールするよう書き換える。

```text
langchain
langchain-openai
openai
python-dotenv
```

- 仮想環境(venv)に移動する
- `pip install -r requirements.txt`
- `pip install langchain-community`
- `pip install langchain-cli`
- .env ファイルを作成して、LLM の API キーを格納

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

Slack に Dify を Slackbot の形で利用  
https://app.slack.com/client/T087FCDUFP0/C087K6KJ39B?ssb_vid=.0a757345d0fade9374201a02344da7ba&ssb_instance_id=2fc413fa-f60c-4fa8-a515-61ce8128e24d

## 環境構築

.env ファイルを作成して環境変数を設定  
python3 -m venv venv  
pip install -r requirements.txt

## デバック

source venv/bin/activate  
python3 connect_dify.py
