from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import asyncio
import logging

# 環境変数の読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

###############
# LLMの呼び出し #
###############
# ChatOpenAIのインスタンスを生成
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

# AIエージェントによるタスク選択
def call_ai_task_selection(llm, user_prompt):
    prompt_template = PromptTemplate(
        template=(
            "ユーザー入力に基づいて実行するべきタスクを選択してください。\n\n"
            "タスクは次の中から選んでください: A, B, C。\n"
            "出力はコンマ区切りで記述してください (例: A, B)。\n\n"
            "ユーザー入力: {user_prompt}\n\n"
            "実行するタスク:"
        ),
        input_variables=["user_prompt"]
    )
    chain = prompt_template | llm
    response = chain.invoke({"user_prompt": user_prompt})
    return response.content.strip().split(", ")

###############
# Stateの設定 #
###############
# Stateクラスの宣言
class State(TypedDict):
    message: str

##############
# Nodeの設定 #
##############
# 開始ノード
def node_start(state: State, config: RunnableConfig):
    logging.info("[DEBUG] node_start executed")
    prompt = state["message"]
    return {"message": prompt}

# タスクAのノード
async def node_task_A(state: State, config: RunnableConfig):
    logging.info("[DEBUG] node_task_A executed")
    await asyncio.sleep(1)  # 処理Aをシミュレート
    return {"message": f"Task A completed. Original Message: {state['message']}"}

# タスクBのノード
async def node_task_B(state: State, config: RunnableConfig):
    logging.info("[DEBUG] node_task_B executed")
    await asyncio.sleep(2)  # 処理Bをシミュレート
    return {"message": f"Task B completed. Original Message: {state['message']}"}

# タスクCのノード
async def node_task_C(state: State, config: RunnableConfig):
    logging.info("[DEBUG] node_task_C executed")
    await asyncio.sleep(3)  # 処理Cをシミュレート
    return {"message": f"Task C completed. Original Message: {state['message']}"}

# AIエージェントによるタスク選択ノード
def node_task_selection(state: State, config: RunnableConfig):
    logging.info("[DEBUG] node_task_selection executed")
    selected_tasks = call_ai_task_selection(llm, state["message"])
    return {"selected_tasks": selected_tasks, "message": state["message"]}

#############
# LangGraph #
#############
# Graphの作成
graph_builder = StateGraph(State)

# ノードの追加
graph_builder.add_node("node_start", node_start)
graph_builder.add_node("node_task_selection", node_task_selection)
graph_builder.add_node("node_task_A", node_task_A)
graph_builder.add_node("node_task_B", node_task_B)
graph_builder.add_node("node_task_C", node_task_C)

# グラフの始点を設定
graph_builder.set_entry_point("node_start")

# タスク選択後のルーティング
def routing(state: State, config: RunnableConfig):
    selected_tasks = state["selected_tasks"]
    if "A" in selected_tasks:
        logging.info("[DEBUG] Routing to node_task_A")
        return "node_task_A"
    elif "B" in selected_tasks:
        logging.info("[DEBUG] Routing to node_task_B")
        return "node_task_B"
    elif "C" in selected_tasks:
        logging.info("[DEBUG] Routing to node_task_C")
        return "node_task_C"
    else:
        logging.warning("[DEBUG] No valid task selected.")
        return None

# タスク選択ノードからのエッジ設定
graph_builder.add_conditional_edges("node_task_selection", routing)

# グラフの終点を設定
graph_builder.set_finish_point("node_task_A")
graph_builder.set_finish_point("node_task_B")
graph_builder.set_finish_point("node_task_C")

# Graphをコンパイル
graph = graph_builder.compile()

##################
# グラフの実行例 #
##################
# 実行例のStateを設定
initial_state = {"message": "AとBを実行して。あとCもやって。"}

# Graphの実行
graph.invoke(initial_state, debug=True)
