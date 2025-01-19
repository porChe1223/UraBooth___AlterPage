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
import matplotlib.pyplot as plt
import chainlit as cl
import requests

#========================
# LLMのAPIキー読み込み
#========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY_1")
openai_api_key2 = os.getenv("OPENAI_API_KEY_2")
openai_api_key3 = os.getenv("OPENAI_API_KEY_3")
openai_api_key4 = os.getenv("OPENAI_API_KEY_4")

#===================
# LLMの呼び出し
#===================
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

#=================
# CosmosDBのURL
#=================
urls = [
    "https://cosmosdbdatagetter.azurewebsites.net/data",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=トラフィックソース関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=ユーザー行動関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=サイト内検索関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=デバイスおよびユーザ属性関連情報",
    "https://cosmosdbdatagetter.azurewebsites.net/data?group=時間帯関連情報",
]


######################
######################
## LangChain上の関数
##
#================
# ユーティリティ 
#================
# テキストファイルの読み込み
def txt_read(txtfile):
    f = open(txtfile, "r")
    data = f.read()
    f.close()
    return data

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

#===============
# 最初の分類
#===============
def select_What_to_do(llm, user_prompt):
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"],
        template = txt_read("resource/sys_prompt/start.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#========================================
# 標準回答
# - マーケティングのスペシャリストとして
#========================================
# 標準回答の関数
def call_Others(llm, user_prompt):
    # Alterboothのシステムプロンプト
    system_prompt_Alterbooth = PromptTemplate(
        input_variables = ['user_prompt'],
        template = txt_read("resource/sys_prompt/specialist.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        system_prompt_Alterbooth
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#===================
# プロンプトの評価
#===================
def call_Review(llm, user_prompt):
    # プロンプト評価用のシステムプロンプト
    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = (
            txt_read("resource/sys_prompt/evaluation.txt") +  
            #txt_read("resource/sys_prompt/grobal.txt") +  
            "\n\nプロンプト: {user_prompt} \n\nプロンプトの評価:"
        )
    )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#===================
# ブログ解析
#===================
#AIエージェントがユーザの質問を分類し、適切なワークフロー（tool01かtool2）を呼び出すための関数の定義
def select_tool(llm, user_prompt):
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

# AIエージェントがルートを分析か、分析＋提案かを選択する関数の定義
def fork_tool(llm, user_prompt):
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"], #route01~route06が選択される
        template = txt_read("resource/sys_prompt/fork.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

# 渡されたデータの分析をする関数
def call_Analyze(llm, user_prompt, url_index=0):
    llm = llm
    url = urls[url_index]
    response = requests.get(url)
    print(response.text)

    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = (
             response.text + 
             txt_read("resource/rag_data/ga4Info.txt")
        ),
        template_format="jinja2"
        )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

# データ分析結果を要約する関数の定義
def call_Summarize(llm, massage):
    # データ取得のシステムプロンプト
    system_prompt_Summarize = PromptTemplate(
        input_variables = ["message"],
        template = (
            txt_read("resource/sys_prompt/summary.txt") + 
            "\n\nプロンプト: {user_prompt}"
        )
    )
    # チェーンの宣言
    chain = (
        system_prompt_Summarize
        | llm
    )
    # チェーンの実行
    return chain.invoke({"message": massage })

# 分析されたデータからの改善策提案の関数
def call_Advice(llm, user_prompt):
    # データ取得のシステムプロンプト
    system_prompt_Analyze = PromptTemplate(
        input_variables = ["user_prompt"],
        template = (
            txt_read("resource/user_prompt/answer.txt") #+ 

            #"\n\nプロンプト: {user_prompt}"
        )
    )
    # チェーンの宣言
    chain = (
        system_prompt_Analyze
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})


################################
################################
## LangGraphによるワークフロー
##
#==================
# Stateの宣言
#==================
class State(TypedDict):
    message: str

#==============
# Nodeの宣言
#==============
# 開始ノード（最初の動作を決める）
def node_Start(state: State, config: RunnableConfig):
     prompt = state["message"]
     if 'プロンプト' in prompt and '評価' in prompt:
        return {"message_type": "<プロンプト評価>", "messages": prompt}
     else:
        response = select_What_to_do(llm4, prompt)
        return {"message_type": response.content, "messages": prompt}

# 標準回答ノード
def node_Others(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Others(llm, prompt)
    return {"message": response.content}

# プロンプトの評価ノード
def node_Review(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Review(llm, prompt)
    return { "message": response.content }

# 分析項目を全体か限定かに分類するノード
def node_Main(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = select_tool(llm, prompt)
    return {"message_type": response.content, "messages": prompt}

# ユーザの質問を細かく分類するノード
def node_Classify(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = classify_tool(llm3, prompt)
    return {"message_type": response.content, "messages": prompt}

# ルートを分析か、分析＋提案かを選択するノード
def node_Fork(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = fork_tool(llm4, prompt)
    return {"message_type": response.content, "messages": prompt}

# 全データ分析ノード
def node_DataGet(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2,prompt,url_index=0)
    return {"message":result_data}

# ページ関連情報分析ノード
def node_DataGet_page(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=1)
    return {"message":result_data}

# トラフィックソース関連情報分析ノード
def node_DataGet_traffic(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=2)
    return {"message":result_data}

# ユーザー関連情報分析ノード
def node_DataGet_user(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=3)
    return {"message":result_data}

# サイト内検索情報分析ノード
def node_DataGet_search(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=4)
    return {"message":result_data}

# デバイスおよびユーザー属性情報分析ノード
def node_DataGet_device(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=5)
    return {"message":result_data}

# 時間帯関連情報分析ノード
def node_DataGet_time(state: State, config: RunnableConfig):
    prompt = state["message"]
    result_data = call_Analyze(llm2, prompt, url_index=6)
    return {"message":result_data}

# データ分析結果を要約するノード
def node_Summarize(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Summarize(llm4, prompt)
    return {"message": f"要約: {response.content}"}

# データ結果を解析するノード
def node_Advice(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Advice(llm, prompt)
    return {"message": f"解析結果: {response.content}"}

# 終了ノード
def node_End(state: State, config: RunnableConfig):
    return {"message": "終了"}

#================
# Graphの作成
#================
graph_builder = StateGraph(State)

#===============
# Nodeの追加
#===============
graph_builder.add_node("node_Start", node_Start)
graph_builder.add_node("node_Others", node_Others)
graph_builder.add_node("node_Review", node_Review)
graph_builder.add_node("node_Main", node_Main)
# 分野別
graph_builder.add_node("node_Classify", node_Classify)
graph_builder.add_node("node_Fork", node_Fork)
graph_builder.add_node("node_DataGet_page", node_DataGet_page)
graph_builder.add_node("node_DataGet_traffic", node_DataGet_traffic)
graph_builder.add_node("node_DataGet_user", node_DataGet_user)
graph_builder.add_node("node_DataGet_search", node_DataGet_search)
graph_builder.add_node("node_DataGet_device", node_DataGet_device)
graph_builder.add_node("node_DataGet_time", node_DataGet_time)
#さらに分岐して分析と提案
graph_builder.add_node("node_toSuggestion_search", node_DataGet_search)

# 全分野
# graph_builder.add_node("node_DataGet", node_DataGet)
graph_builder.add_node("node_toData_page", node_DataGet_page)
graph_builder.add_node("node_toData_traffic", node_DataGet_traffic)
graph_builder.add_node("node_toData_user", node_DataGet_user)
graph_builder.add_node("node_toData_search", node_DataGet_search)
graph_builder.add_node("node_toData_device", node_DataGet_device)
graph_builder.add_node("node_toData_time", node_DataGet_time)

graph_builder.add_node("node_Summarize", node_Summarize)
graph_builder.add_node("node_Advice", node_Advice)
graph_builder.add_node("node_End", node_End)

#===================
# ワークフロー作成
#===================
# Graphの始点
graph_builder.set_entry_point("node_Start")

# 始点　＝＞　標準回答 or プロンプト評価 or ブログ解析
graph_builder.add_conditional_edges(
    "node_Start",
    lambda state: state["message_type"],
    {
        "<回答>": "node_Others",
        "<プロンプト評価>": "node_Review",
        "<ブログ解析>": "node_Main",
    },
)
# # 条件分岐のエッジを追加
# graph_builder.add_conditional_edges( # 条件分岐のエッジを追加
#     'node_Start',
#     routing, # 作ったルーティング
# )

# 標準回答　＝＞　終点
graph_builder.add_edge("node_Others", "node_End")

# プロンプト評価　＝＞　終点
graph_builder.add_edge("node_Review", "node_End")

# 全体 or グループを限定
graph_builder.add_conditional_edges(
    "node_Main",
    lambda state: state["message_type"],
    {
        "tool01": "node_Classify",
        "tool02": "node_toData_page",
    },
)

# 6つのディメンショングループ
graph_builder.add_conditional_edges(
    "node_Classify",
    lambda state: state["message_type"],
    {
        "route01": "node_DataGet_page",
        "route02": "node_DataGet_traffic",
        "route03": "node_DataGet_user",
        "route04": "node_Fork",
        "route05": "node_DataGet_device",
        "route06": "node_DataGet_time",
    },
)

# サイト内関連情報を解析 or 解析＋提案
graph_builder.add_conditional_edges(
    "node_Fork",
    lambda state: state["message_type"],
    {
        "only_analyze": "node_DataGet_search",
        "analyze_to_suggestion": "node_toSuggestion_search",
    },
)

"""
# 分野別分析
# ページ関連情報分析　＝＞　解析
graph_builder.add_edge("node_DataGet_page", "node_Advice")
# トラフィックソース関連情報分析　＝＞　解析
graph_builder.add_edge("node_DataGet_traffic", "node_Advice")
# ユーザー関連情報分析　＝＞　解析
graph_builder.add_edge("node_DataGet_user", "node_Advice")
# サイト内検索情報分析　＝＞　解析
#graph_builder.add_edge("node_DataGet_search", "node_Advice")
# デバイスおよびユーザー属性情報分析　＝＞　解析
graph_builder.add_edge("node_DataGet_device", "node_Advice")
# 時間帯関連情報分析　＝＞　解析
graph_builder.add_edge("node_DataGet_time", "node_Advice")
"""

# 分野別分析＋提案
# サイト内検索情報分析　＝＞　提案
graph_builder.add_edge("node_toSuggestion_search", "node_Advice")

# 全分野分析
# # 全体データ取得　＝＞　解析
# graph_builder.add_edge("node_DataGet", "node_Advice")
# ページ関連情報分析　＝＞　トラフィックソース関連情報分析
graph_builder.add_edge("node_toData_page", "node_toData_traffic")
# トラフィックソース関連情報分析　＝＞　ユーザー関連情報分析
graph_builder.add_edge("node_toData_traffic", "node_toData_user")
# ユーザー関連情報分析　＝＞　サイト内検索情報分析
graph_builder.add_edge("node_toData_user", "node_toData_search")
# サイト内検索情報分析　＝＞　デバイスおよびユーザー属性情報分析
graph_builder.add_edge("node_toData_search", "node_toData_device")
# デバイスおよびユーザー属性情報分析　＝＞　時間帯関連情報分析
graph_builder.add_edge("node_toData_device", "node_toData_time")
# 時間帯関連情報分析　＝＞　要約
graph_builder.add_edge("node_toData_time", "node_Summarize")
# 要約　＝＞　解析
graph_builder.add_edge("node_Summarize", "node_Advice")

# 解析　＝＞　終点
graph_builder.add_edge("node_Advice", "node_End")
graph_builder.add_edge("node_DataGet_search", "node_End")

# 終点
graph_builder.add_edge("node_DataGet_page", "node_End")
graph_builder.add_edge("node_DataGet_traffic", "node_End")
graph_builder.add_edge("node_DataGet_user", "node_End")
graph_builder.add_edge("node_DataGet_device", "node_End")
graph_builder.add_edge("node_DataGet_time", "node_End")
graph_builder.add_edge("node_toSuggestion_search", "node_End")



# Graphをコンパイル
graph = graph_builder.compile()


########################
########################
## Chainlitでアプリ化
##
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