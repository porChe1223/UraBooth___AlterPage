import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import requests
from dotenv import load_dotenv

load_dotenv()

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# ボットのユーザーIDを取得
bot_user_id = app.client.auth_test()["user_id"]

@app.event("app_mention")
def handle_app_mention(event, say):
    print(event)
    if event and 'text' in event:
        dify_api_key = os.environ["DIFY_API_KEY"]
        url = 'http://localhost/v1/chat-messages'  # Dify API endpoint
        user = event['user']
        query = event['text'].replace(f"<@{bot_user_id}>", "").strip() # メンション部分を削除
        headers = {
            'Authorization': f'Bearer {dify_api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'query': query,
            'response_mode': 'blocking',
            'user': user,
            'conversation_id': '', # 必要に応じて設定
            'inputs': {}
        }
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        if 'answer' in response_data:
            say(response_data['answer'])
        else:
            say(f"Dify APIからの予期しないレスポンス: {response_data}")
    else:
        say("メッセージの内容を取得できませんでした。")

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()