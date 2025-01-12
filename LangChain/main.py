from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import openai
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
import os
import time
import random
import matplotlib.pyplot as plt

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Stateの宣言
class State(TypedDict):
    value: str



################
# LLMの呼び出し #
################
# LLMの宣言
llm = ChatOpenAI(model_name="gpt-4o-mini",
                 temperature=0.5,
                 openai_api_key=openai_api_key
                 )

# プロンプトの宣言
human_message_prompt = HumanMessagePromptTemplate(
    prompt = PromptTemplate(
        input_variables=["job"],
        template="{job}に一番オススメのプログラミング言語は何?"
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

# チェーンの宣言
chain = LLMChain(
    llm=llm,
    prompt=chat_prompt_template
)

# チェーンの実行関数
def exec_chain(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # LangChainチェーンを実行
        time.sleep(delay)
        return chain.invoke({"job": prompt})  # invokeを使うことでチェーンを実行可



############
# 実行関数 #
############
def exec_func_A(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"AAA: {prompt}"}
    
def exec_func_B(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"BBB: {prompt}"}

def exec_func_C(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"CCC: {prompt}"}



##############
# Nodeの宣言 #
##############
# 開始ノード
def start_node(state: State, config: RunnableConfig):
    prompt = state["value"]                   # LLMに渡すプロンプト
    # response = retry_requestA(None, prompt) # LLMの呼び出し
    # return {"value": response['content']}   # 応答を新しいStateに格納
    return {"value": f"開始: {prompt}"}

# ブログの改善ノード
def node_A1(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)
    return {"value": response['content']}

def node_A2(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)
    return {"value": response['content']}

def node_A3(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)
    return {"value": response['content']}

def node_A4(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)
    return {"value": response['content']}

def node_A5(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)

    data = [1, 2, 3, 4, 5]
    plt.plot(data)
    plt.title('Sample Graph')
    plt.xlabel('X-axis')                  # グラフ描画の処理
    plt.ylabel('Y-axis')
    plt.savefig('graph.png') # グラフの処理方法が悩みどころ

    return {"value": response['content']}

def node_A6(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_A(None, prompt)
    return {"value": response['content']}


# プロンプトの評価ノード
def node_B1(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_B(None, prompt)
    return {"value": response['content']}


# その他ノード
def node_C1(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_C(None, prompt)
    return {"value": response['content']}

def node_C2(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_func_C(None, prompt)
    return {"value": response['content']}


# 終了ノード
def end_node(state: State, config: RunnableConfig):
    prompt = state["value"]
    return {"value": f"終了: {prompt}"}



#############
# LangGraph #
#############
# Graphの作成
graph_builder = StateGraph(State)

# Nodeの追加
graph_builder.add_node("start_node", start_node)
graph_builder.add_node("node_A1", node_A1)
graph_builder.add_node("node_A2", node_A2)
graph_builder.add_node("node_A3", node_A3)
graph_builder.add_node("node_A4", node_A4)
graph_builder.add_node("node_A5", node_A5)
graph_builder.add_node("node_A6", node_A6)
graph_builder.add_node("node_B1", node_B1)
graph_builder.add_node("node_C1", node_C1)
graph_builder.add_node("node_C2", node_C2)
graph_builder.add_node("end_node", end_node)

# Graphの始点を宣言
graph_builder.set_entry_point("start_node")

# ルーティングの設定
def routing(state: State, config: RunnableConfig) -> Literal["node_A1", "node_B1", "node_C1"]:
    random_num = random.randint(0, 2)
    if random_num == 0:
        return "node_A1"
    elif random_num == 1:
        return "node_B1"
    else:
        return "node_C1"

# 条件分岐のエッジを追加
graph_builder.add_conditional_edges( # 条件分岐のエッジを追加
    'start_node',
    routing, # 作ったルーティング
)

# Nodeをedgeに追加
graph_builder.add_edge("node_A1", "node_A2")
graph_builder.add_edge("node_A2", "node_A3")
graph_builder.add_edge("node_A3", "node_A4")
graph_builder.add_edge("node_A4", "node_A5")
graph_builder.add_edge("node_A5", "node_A6")
graph_builder.add_edge("node_C1", "node_C2")

# Graphの終点を宣言
graph_builder.set_finish_point("end_node")

# Graphをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)
graph.invoke({"value": "こんにちは"}, debug=True)