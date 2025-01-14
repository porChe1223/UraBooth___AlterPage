print("[DEBUG] Starting test3.py")

from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
import openai
from langchain.prompts import PromptTemplate
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
llm = ChatOpenAI(model_name="gpt-4o-mini", 
                 temperature=0, 
                 openai_api_key=openai_api_key
                 )
print(f"[DEBUG] LLM configured with model: {llm.model_name}")
# AIエージェントによる分類タスク選択
def classification(llm, user_prompt):
    prompt_template = PromptTemplate(
        input_variables=["user_prompt"],
        template=(
            "ユーザー入力が以下の分類基準に該当する場合、すべて該当する分類結果を列挙してください：\n"
            "1. 入力に 'A', 'B', 'C', 'D', 'E', 'F', 'G' のいずれかが含まれる場合は '処理1'\n"
            "2. それ以外の場合は '処理2'\n"
            "\n"
            "ユーザー入力: {user_prompt}\n"
            "分類結果（例: '処理1, 処理2'）:"
        ),
    )

    # チェーンの宣言
    chain = prompt_template | llm
    # デバッグ用にプロンプトを出力
    prompt = prompt_template.format(user_prompt=user_prompt)
    print(f"[DEBUG] Prompt sent to OpenAI:\n{prompt}")
    # チェーンの実行
    #return chain.invoke({"user_prompt": user_prompt})
    result = chain.invoke({"user_prompt": user_prompt})
    print(f"[DEBUG] OpenAI response:\n{result}")
    return result

###############
# Stateの設定 #
###############
class State(TypedDict):
    message: str
    classifications: list[str]  # 複数の分類結果を保持

##############
# Nodeの設定 #
##############
def node_start(state: State, config: RunnableConfig):
    return {"message": state["message"]}

async def node_classification(state: State, config: RunnableConfig):
    try:
        prompt = state["message"]
        classification_result = await classification(llm, prompt)
        print(f"[DEBUG] Classification result:\n{classification_result}")
    # LLMの出力を分割して分類結果をリストに変換
        classifications = [
            result.strip() for result in classification_result.content.split(",") if result.strip()
        ]
        return {"classifications": classifications, "message": prompt}
    except Exception as e:
        print(f"[ERROR] Exception in node_classification: {e}")
        raise

async def node_process_1(state: State, config: RunnableConfig):
    prompt = state["classifications"]
    await asyncio.sleep(1)  # 処理1をシミュレート
    return {"message": f"処理1が完了しました。"}

async def node_process_2(state: State, config: RunnableConfig):
    prompt = state["classifications"]
    await asyncio.sleep(1)  # 処理2をシミュレート
    return {"message": f"処理2が完了しました。"}

#############
# LangGraph #
#############
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
    classifications = state.get("classifications", [])
    
    # 分岐処理
    if "処理1" in classifications and "処理2" in classifications:
        logging.info("[DEBUG] Routing to both node_process_1 and node_process_2")
        return ["node_process_1", "node_process_2"]
    elif "処理1" in classifications:
        logging.info("[DEBUG] Routing to node_process_1")
        return "node_process_1"
    elif "処理2" in classifications:
        logging.info("[DEBUG] Routing to node_process_2")
        return "node_process_2"
    else:
        logging.warning("[DEBUG] No valid classification. Defaulting to node_process_2.")
        return "node_process_2"

# 分類ノードからのエッジ設定
graph_builder.add_conditional_edges(
    "node_classification",
    routing,
)

# グラフの終点を設定
graph_builder.set_finish_point("node_process_1")
graph_builder.set_finish_point("node_process_2")

# Graphをコンパイル
graph = graph_builder.compile()

##################
# グラフの実行例 #
##################
# Graphの実行
graph.invoke({"message": "A, H", "classifications": []}, debug=True)
