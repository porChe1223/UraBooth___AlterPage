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
def exec_Normal(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"NNN: {prompt}"}

def exec_Review(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"RRR: {prompt}"}

def exec_Analyze(chain, prompt, retries=3, delay=30):
    for attempt in range(retries):
        # モックデータを返す
        return {"content": f"AAA: {prompt}"}



##############
# Nodeの宣言 #
##############
# 開始ノード
def node_Start(state: State, config: RunnableConfig):
    prompt = state["value"]                   # LLMに渡すプロンプト
    # response = retry_requestA(None, prompt) # LLMの呼び出し
    # return {"value": response['content']}   # 応答を新しいStateに格納
    return {"value": f"開始: {prompt}"}

# 標準回答(アプリの説明付き)ノード
def node_Normal(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_Normal(None, prompt)
    return {"value": response['content']}

# プロンプトの評価ノード
def node_Review(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_Review(None, prompt)
    return {"value": response['content']}

# ブログの改善ノード
def node_Analyze(state: State, config: RunnableConfig):
    prompt = state["value"]
    response = exec_Analyze(None, prompt)
    return {"value": response['content']}

# def node_A1(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)
#     return {"value": response['content']}

# def node_A2(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)
#     return {"value": response['content']}

# def node_A3(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)
#     return {"value": response['content']}

# def node_A4(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)
#     return {"value": response['content']}

# def node_A5(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)

#     data = [1, 2, 3, 4, 5]
#     plt.plot(data)
#     plt.title('Sample Graph')
#     plt.xlabel('X-axis')                  # グラフ描画の処理
#     plt.ylabel('Y-axis')
#     plt.savefig('graph.png') # グラフの処理方法が悩みどころ

#     return {"value": response['content']}

# def node_A6(state: State, config: RunnableConfig):
#     prompt = state["value"]
#     response = exec_func_A(None, prompt)
#     return {"value": response['content']}


# 終了ノード
def node_End(state: State, config: RunnableConfig):
    prompt = state["value"]
    return {"value": f"終了: {prompt}"}



#############
# LangGraph #
#############
# Graphの作成
graph_builder = StateGraph(State)

# Nodeの追加
graph_builder.add_node("node_Start", node_Start)
graph_builder.add_node("node_Normal", node_Normal)
graph_builder.add_node("node_Review", node_Review)
graph_builder.add_node("node_Analyze", node_Analyze)
# graph_builder.add_node("node_A1", node_A1)
# graph_builder.add_node("node_A2", node_A2)
# graph_builder.add_node("node_A3", node_A3)
# graph_builder.add_node("node_A4", node_A4)
# graph_builder.add_node("node_A5", node_A5)
# graph_builder.add_node("node_A6", node_A6)
graph_builder.add_node("node_End", node_End)

# Graphの始点を宣言
graph_builder.set_entry_point("node_Start")

# ルーティングの設定
def routing(state: State, config: RunnableConfig) -> Literal["node_Normal", "node_Review", "node_Analyze"]:
    random_num = random.randint(0, 2)
    if random_num == 0:
        return "node_Normal"
    elif random_num == 1:
        return "node_Review"
    else:
        return "node_Analyze"

# 条件分岐のエッジを追加
graph_builder.add_conditional_edges( # 条件分岐のエッジを追加
    'node_Start',
    routing, # 作ったルーティング
)

# Nodeをedgeに追加
# graph_builder.add_edge("node_Analyze", "node_A1")
# graph_builder.add_edge("node_A1", "node_A2")
# graph_builder.add_edge("node_A2", "node_A3")
# graph_builder.add_edge("node_A3", "node_A4")
# graph_builder.add_edge("node_A4", "node_A5")
# graph_builder.add_edge("node_A5", "node_A6")
# graph_builder.add_edge("node_C1", "node_C2")

# Graphの終点を宣言
graph_builder.set_finish_point("node_End")

# Graphをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)
graph.invoke({"value": "こんにちは"}, debug=True)


# # システムプロンプトを読み込む
# def load_system_prompt(file_path):
#     p = open(file_path, "r")    
#     system_prompt = p.read()
#     p.close()
#     return system_prompt

# # システムプロンプトファイルを読み込む
# system_prompt = load_system_prompt("evaluationprompt.txt")

# # プロンプト評価のためのテンプレート
# evaluation_prompt = PromptTemplate(
#     input_variables=["user_prompt"],
#     template=system_prompt + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
# )

# # チェーンを定義
# general_chain = LLMChain(llm=llm, prompt=prompt)
# evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

# def process_input(user_input: str):
#     # 入力文に「評価」という単語が含まれているかを確認
#     if "プロンプト" in user_input and "評価" in user_input:
#         print("プロンプトを評価します。")
#         # プロンプトの評価・改善を実行
#         response = evaluation_chain.run(user_prompt=user_input)
#     else:
#         # 通常の処理を実行
#         response = general_chain.run(job=user_input)
    
#     return response

# # メイン処理
# if __name__ == "__main__":
#     # ユーザー入力（例）
#     user_input = "データサイエンティストが直面する典型的なプロジェクトに基づいて、最も適したプログラミング言語を3つ挙げ、それぞれの言語の利点を具体的に説明してください。特に、言語の人気、学習の容易さ、コミュニティのサポートの観点から評価してください。プロンプトの評価を行なって"
    
#     # 処理を実行
#     output = process_input(user_input)
#     print(output)
