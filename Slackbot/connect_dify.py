import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests
from dotenv import load_dotenv
import json

load_dotenv()

URL_PATH = 'https://api.dify.ai/v1/chat-messages'

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# ボットのユーザーIDを取得
bot_user_id = app.client.auth_test()["user_id"]

@app.event("app_mention")
def handle_app_mention(event, say):
    print(event)
    if event and 'text' in event:
        dify_api_key = os.environ["DIFY_API_KEY"]
        url = URL_PATH  # Dify API endpoint
        user = event['user']
        query = event['text'].replace(f"<@{bot_user_id}>", "").strip() # メンション部分を削除
        headers = {
            'Authorization': f'Bearer {dify_api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "query": query,
            'response_mode': 'blocking',
            "user": user,  # userパラメータを追加
            'conversation_id': '', # 必要に応じて設定
            'inputs': {}
        }
        print(f'リクエスト送信先: {url}')
        print(f'ヘッダー: {headers}')
        print(f'返答結果: {data}')
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            try:
                response_data = response.json()
                # レスポンスデータの処理
                if 'answer' in response_data:
                    say(response_data['answer'])
                else:
                    say(f"Dify APIからの予期しないレスポンス: {response_data}")
            except json.JSONDecodeError:
                print("JSONデコードエラー: レスポンスが空です")
                response_data = None
                say("Dify APIからのレスポンスが空です")
        else:
            print(f"リクエスト失敗: ステータスコード {response.status_code}")
            print(f"レスポンス内容: {response.text}")
            response_data = None
            say("Dify APIへのリクエストが失敗しました")

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()