from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import openai
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import Document
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import time
import random
import matplotlib.pyplot as plt
import langchain.vectorstores
import logging
import chainlit as cl
import requests

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key2 = os.getenv("OPENAI_API_KEY2")
openai_api_key3 = os.getenv("OPENAI_API_KEY3")
openai_api_key4 = os.getenv("OPENAI_API_KEY4")



################
# LLMの呼び出し #
################
# LLMの宣言
llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0,
                 openai_api_key=openai_api_key
                 )
llm2 = ChatOpenAI(model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_api_key2
                    )
llm3 = ChatOpenAI(model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_api_key3
                    )
llm4 = ChatOpenAI(model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_api_key4
                    )

# プロンプトに対してそのまま返答
def call_llm(llm, prompt):
    # プロンプトの宣言
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=[]
    )
    # チェーンの宣言
    chain = (
        prompt_template
        | llm
    )
    return chain.invoke({})

# テキストファイルの読み込み
def txt_read(txtfile):
    f = open(txtfile, "r")
    data = f.read()
    f.close()
    return data

""""""""""""""""""""
" ブログの改善提案   "
" ここに関数を入れる "
""""""""""""""""""""

#AIエージェントがユーザの質問を分類し、適切なワークフロー（tool01かtool2）を呼び出すための関数の定義

def select_tool(llm, user_prompt):
    llm = llm
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"], #tool1かtool2が選択される
        template = txt_read("resource/sys_prompt/classification.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#AIエージェントがユーザの質問を分類し、適切なワークフロー（route01~route06）を呼び出すための関数の定義
def classify_tool(llm, user_prompt):
    llm = llm3
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"], #route01~route06が選択される
        template = txt_read("resource/sys_prompt/specific_classify.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})



# プロンプトの評価
def call_Review(llm, user_prompt):
    # プロンプト評価用のシステムプロンプト
    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = txt_read("resource/sys_prompt/evaluation.txt") + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
    )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

# その他の標準回答 <- Alterboothのプロフェッショナルとして
def call_Others(llm, user_prompt):
    # Alterboothのシステムプロンプト
    system_prompt_Alterbooth = PromptTemplate(
        input_variables = ['user_prompt'],
        template = txt_read("resource/rag_data/alterbooth.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        system_prompt_Alterbooth
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

# 渡されたデータの分析をする関数の定義
def call_Advice(llm, user_prompt):
    llm = llm
    # データ取得のシステムプロンプト
    system_prompt_Analyze = PromptTemplate(
        input_variables = ["user_prompt"],
        template = (
            txt_read("resource/rag_data/ga4Info.txt") + 
            txt_read("resource/rag_data/alterbooth.txt") + 
            #txt_read("resource/sys_prompt/grobal.txt") +  
            "\n\nプロンプト: {user_prompt}"
        )
    )
    # チェーンの宣言
    chain = (
        system_prompt_Analyze
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})



###############
# Stateの宣言 #
###############
class State(TypedDict):
    message: str

##############
# Nodeの宣言 #
##############
# 開始ノード
def node_Start(state: State, config: RunnableConfig):
    # 入力されたメッセージをそのまま次に渡す
    return {"message": state["message"]}

# ブログの改善ノード
def node_Main(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = select_tool(llm, prompt)
    #logging.info(f"[DEBUG] response: {response}")
    #logging.info(f"[DEBUG] response.content: {response.content}")
    return {"message_type": response.content, "messages": prompt}

# プロンプトの評価ノード
def node_Review(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Review(llm, prompt)
    return { "message": response.content }

# その他の標準回答ノード
def node_Others(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Others(llm, prompt)
    return {"message": response.content}

# ユーザの質問を細かく分類するノード
def node_Classify(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = classify_tool(llm3, prompt)
    return {"message_type": response.content, "messages": prompt}


############## ここにデータ取得の関数を追加する ####################

# データ取得関連のURL
urls = [
    "https://cosmosdbdatagetter.azurewebsites.net/data",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=トラフィックソース関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=ユーザー行動関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=サイト内検索関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=デバイスおよびユーザ属性関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=時間帯関連情報",
    ]

# データ取得関数
def call_Analyze(llm, user_prompt, url_index=0):
    llm = llm
    url = urls[url_index]
    #url = "https://～.azurewebsites.net/data?group=ページ関連情報"
    response = requests.get(url)
    print(response.text)

    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template =  response.text,
        template_format="jinja2"
        )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})


# 全データ取得ノード
def node_DataGet(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2,prompt,url_index=0)
    return {"message":result_data}

# ページ関連情報取得ノード
def node_DataGet_page(state: State, config: RunnableConfig):
    prompt = state["message"]
    #print(prompt)
    result_data = call_Analyze(llm2, prompt, url_index=1)
    #print(result_data)
    return {"message":result_data}

# トラフィックソース関連情報取得ノード
def node_DataGet_traffic(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=2)
    return {"message":result_data}

# ユーザー関連情報取得ノード
def node_DataGet_user(state: State, config: RunnableConfig):
    #if url is None:
    #    url = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=ユーザー関連情報"]
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=3)
    return {"message":result_data}

# サイト内検索情報取得ノード
def node_DataGet_search(state: State, config: RunnableConfig):
    #if url is None:
    #    url = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=サイト内検索関連情報"]
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=4)
    return {"message":result_data}

# デバイスおよびユーザー属性情報取得ノード
def node_DataGet_device(state: State, config: RunnableConfig):
    #if url is None:
    #    url = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=デバイスおよびユーザー属性関連情報"]
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=5)
    return {"message":result_data}

# 時間帯関連情報取得ノード
def node_DataGet_time(state: State, config: RunnableConfig):
    #if url is None:
    #    url = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=時間帯関連情報"]
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=6)
    return {"message":result_data}

####################################################

# データ結果を解析するノード
def node_Advice(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Advice(llm, prompt)
    return {"message": f"解析結果: {response.content}"}

# プロンプトの評価ノード
def node_Review(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Review(llm, prompt)
    return { "message": response.content }


# 終了ノード
def node_End(state: State, config: RunnableConfig):
    return {"message": "終了"}



#############
# LangGraph #
#############
# Graphの作成
graph_builder = StateGraph(State)

# Nodeの追加
graph_builder.add_node("node_Start", node_Start)
graph_builder.add_node("node_Main", node_Main)
graph_builder.add_node("node_Review", node_Review)
graph_builder.add_node("node_Others", node_Others)
graph_builder.add_node("node_Classify", node_Classify)
graph_builder.add_node("node_DataGet", node_DataGet)
graph_builder.add_node("node_DataGet_page", node_DataGet_page)
graph_builder.add_node("node_DataGet_traffic", node_DataGet_traffic)
graph_builder.add_node("node_DataGet_user", node_DataGet_user)
graph_builder.add_node("node_DataGet_search", node_DataGet_search)
graph_builder.add_node("node_DataGet_device", node_DataGet_device)
graph_builder.add_node("node_DataGet_time", node_DataGet_time)
graph_builder.add_node("node_Advice", node_Advice)
graph_builder.add_node("node_End", node_End)

# Graphの始点を宣言
graph_builder.set_entry_point("node_Start")

# ルーティングの設定
def routing(state: State, config: RunnableConfig):
    if 'ブログ' in state['message']:
        logging.info(f"[DEBUG] Routing to node_Main")
        return "node_Main"
    
    elif 'プロンプト' in state['message'] and '評価' in state['message']:
        logging.info(f"[DEBUG] Routing to node_Review")
        return "node_Review"
    
    else:
        logging.info(f"[DEBUG] Routing to node_Others")
        return "node_Others"

# 条件分岐のエッジを追加
graph_builder.add_conditional_edges( # 条件分岐のエッジを追加
    'node_Start',
    routing, # 作ったルーティング
)


graph_builder.add_conditional_edges(
    "node_Main",
    lambda state: state["message_type"],
    {
        "tool01": "node_Classify",
        "tool02": "node_DataGet",
    },
)



graph_builder.add_conditional_edges(
    "node_Classify",
    lambda state: state["message_type"],
    {
        "route01": "node_DataGet_page",
        "route02": "node_DataGet_traffic",
        "route03": "node_DataGet_user",
        "route04": "node_DataGet_search",
        "route05": "node_DataGet_device",
        "route06": "node_DataGet_time",
    },
)


# Nodeをedgeに追加
graph_builder.add_edge("node_DataGet", "node_Advice")
graph_builder.add_edge("node_DataGet_page", "node_Advice")
graph_builder.add_edge("node_DataGet_traffic", "node_Advice")
graph_builder.add_edge("node_DataGet_user", "node_Advice")
graph_builder.add_edge("node_DataGet_search", "node_Advice")
graph_builder.add_edge("node_DataGet_device", "node_Advice")
graph_builder.add_edge("node_DataGet_time", "node_Advice")
graph_builder.add_edge("node_Advice", "node_End")

graph_builder.add_edge("node_Review", "node_End")
graph_builder.add_edge("node_Others", "node_End")

#graph_builder.add_edge("node_Main", "node_DataGet_search") # 仮のエッジ

# Graphの終点を宣言
# graph_builder.set_finish_point("node_Main")
# graph_builder.set_finish_point("node_Review")
# graph_builder.set_finish_point("node_Others")

# Graphをコンパイル
graph = graph_builder.compile()

# # Graphの実行(引数にはStateの初期値を渡す)
# graph.invoke({'message': 'このプロンプトを評価して。「こんにちは」'}, debug=True)


# アプリコード
@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    stream_generator = graph.stream(
        {"message": msg.content},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config)
    )
    
    # ストリームを受け取りつつ順次出力
    for streamed_msg, metadata in stream_generator:
        # ユーザー以外のメッセージ(AIメッセージなど)があれば送る
        if streamed_msg.content and not isinstance(streamed_msg, HumanMessage):
            # 部分的に出す場合は stream_token
            await final_answer.stream_token(streamed_msg.content)

            # node_End のノードならそこで処理打ち切り
            if metadata.get("name") == "node_End":
                break

    # 全出力が完了したので、Chainlit上にメッセージを確定表示
    await final_answer.send()