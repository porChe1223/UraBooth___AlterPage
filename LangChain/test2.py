from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
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
import asyncio
import logging

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

###############
# LLMの呼び出し #
###############
# ChatOpenAIのインスタンスを生成
llm = ChatOpenAI(model_name="gpt-4o-mini", 
                 temperature=0, 
                 openai_api_key=openai_api_key
                 )

# AIエージェントによる分類タスク選択
def classification(llm, user_prompt):
    prompt_template = PromptTemplate(
        input_variables=["user_prompt"],
        template=(
            "以下の入力を基に、分類してください。\n"
            "分類基準:\n"
            "1. 入力に 'A', 'B', 'C', 'D', 'E', 'F', 'G' のいずれかが含まれる場合は '処理1' に分類\n"
            "2. それ以外の場合は '処理2' に分類\n"
            "\n"
            "ユーザー入力: {user_prompt}\n"
            "分類結果 '処理1' または '処理2':"
        ),
    )
    #prompt = prompt_template.format(user_prompt=user_prompt)
    #response = llm.predict(prompt)
    # 応答を解析して結果を返す
    #return response.strip()

    # チェーンの宣言
    chain = (
        prompt_template
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})


###############
# Stateの設定 #
###############
# Stateクラスの宣言
class State(TypedDict):
    message: str
    classification: str

##############
# Nodeの設定 #
##############
# 開始ノード
def node_start(state: State, config: RunnableConfig):
    prompt = state["message"]
    return {"message": prompt}

# 分類ノード
def node_classification(state: State, config: RunnableConfig):
    prompt = state["message"]
    classification_result = classification(llm, state["message"])
    return {"classification": classification_result.content, "message": state["message"]}

# 処理1ノード
async def node_process_1(state: State, config: RunnableConfig):
    await asyncio.sleep(1)  # 処理1をシミュレート
    return {"message": f"処理1が完了しました。元のメッセージ: {state['message']}"}

# 処理2ノード
async def node_process_2(state: State, config: RunnableConfig):
    await asyncio.sleep(1)  # 処理2をシミュレート
    return {"message": f"処理2が完了しました。元のメッセージ: {state['message']}"}

#############
# LangGraph #
#############
# Graphの作成
graph_builder = StateGraph(State)

# ノードの追加
graph_builder.add_node("node_start", node_start)
graph_builder.add_node("node_classification", node_classification)
graph_builder.add_node("node_process_1", node_process_1)
graph_builder.add_node("node_process_2", node_process_2)

# グラフの始点を設定
graph_builder.set_entry_point("node_start")

# 分類結果によるルーティング
def routing(state: State, config: RunnableConfig):
    classification = state.get("classification", "")
    if classification == "処理1":
        logging.info("[DEBUG] Routing to node_process_1")
        return "node_process_1"
    elif classification == "処理2":
        logging.info("[DEBUG] Routing to node_process_2")
        return "node_process_2"
    else:
        logging.warning("[DEBUG] No valid classification. Defaulting to node_process_2.")
        return "node_process_2"

# 分類ノードからのエッジ設定
graph_builder.add_conditional_edges(
    "node_classification", 
    routing
    )

# グラフの終点を設定
graph_builder.set_finish_point("node_process_1")
graph_builder.set_finish_point("node_process_2")

# Graphをコンパイル
graph = graph_builder.compile()

##################
# グラフの実行例 #
##################
# 実行例のStateを設定
initial_state = {"message": "AとEについて教えてください。", "classification": ""}

# Graphの実行
graph.invoke(initial_state, debug=True)
