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
from dotenv import load_dotenv
import os
import time
import random
import matplotlib.pyplot as plt
import logging
import chainlit as cl

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


################
# LLMの呼び出し #
################
# LLMの宣言
llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0,
                 openai_api_key=openai_api_key
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
def call_Analyze(llm, user_prompt):
    # プロンプト評価用のシステムプロンプト
    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = txt_read("RAG_GAInfo.txt") + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
    )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

# プロンプトの評価
def call_Review(llm, user_prompt):
    # プロンプト評価用のシステムプロンプト
    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = txt_read("RAG_evaluationprompt.txt") + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
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
        template = txt_read("RAG_Alterbooth.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        system_prompt_Alterbooth
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
def node_Analyze(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = call_Analyze(llm, prompt)
    logging.info(f"[DEBUG] response: {response}")
    logging.info(f"[DEBUG] response.content: {response.content}")
    return {"message": response.content}

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
graph_builder.add_node("node_Analyze", node_Analyze)
graph_builder.add_node("node_Review", node_Review)
graph_builder.add_node("node_Others", node_Others)
graph_builder.add_node("node_End", node_End)

# Graphの始点を宣言
graph_builder.set_entry_point("node_Start")

# ルーティングの設定
def routing(state: State, config: RunnableConfig):
    if 'ブログ' in state['message']:
        logging.info(f"[DEBUG] Routing to node_Analyze")
        return "node_Analyze"
    
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

# Nodeをedgeに追加
graph_builder.add_edge("node_Analyze", "node_End")
graph_builder.add_edge("node_Review", "node_End")
graph_builder.add_edge("node_Others", "node_End")

# Graphの終点を宣言
# graph_builder.set_finish_point("node_Analyze")
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