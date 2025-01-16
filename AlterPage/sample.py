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
llm = ChatOpenAI(model_name="gpt-4o",
                 temperature=0,
                 openai_api_key=openai_api_key
                 )
import pandas as pd

def summarize_large_data(data, max_rows=10):
    """
    大きなデータを要約する関数。
    """
    # DataFrameに変換
    df = pd.DataFrame(data["data"], columns=data["columns"])
    
    # 基本統計量を計算
    summary_stats = df.describe(include='all').to_dict()

    # ランダムサンプリング
    sampled_data = df.sample(n=min(max_rows, len(df)), random_state=42).to_dict(orient='records')
    
    return {
        "summary": summary_stats,
        "sampled_data": sampled_data
    }

classify_prompt = PromptTemplate(
    input_variables=["summary", "sample"],
    template=(
        "You are given a dataset summary and a sample of the data. Based on this information, "
        "determine the best way to visualize the data. Provide the following configuration in JSON format:\n"
        "- `graph_type`: The type of graph (e.g., 'bar', 'line', 'scatter').\n"
        "- `x_axis`: The column to use for the X-axis.\n"
        "- `y_axes`: A list of columns to plot on the Y-axis.\n"
        "- `title`: The title of the graph.\n"
        "- `xlabel`: The label for the X-axis.\n"
        "- `ylabel`: The label for the Y-axis.\n\n"
        "Dataset Summary: {summary}\n\n"
        "Sample Data: {sample}"
    )
)

# LangGraphチェーンの構成
class DynamicGraphNode(RunnableNode):
    def invoke(self, state: State):
        # データの要約
        summarized_data = summarize_large_data(state["data"])
        state["summary"] = summarized_data["summary"]
        state["sample"] = summarized_data["sampled_data"]
        
        # LLMに適切なグラフ設定をリクエスト
        llm_config = classify_prompt.invoke({
            "summary": state["summary"],
            "sample": state["sample"]
        })
        
        # グラフ生成
        graph_config = json.loads(llm_config)
        generate_dynamic_graph(state["data"], graph_config)
        
        # 状態を更新
        state["message"] = "Dynamic graph generated and saved as 'dynamic_graph_output.png'"
        return state
