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
import os
import time
import random
import matplotlib.pyplot as plt
import logging
import requests


import matplotlib.pyplot as plt
import pandas as pd
#from langchain_core.chains import RunnableChain
#from langgraph.core import State, RunnableNode
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import json
import os
from langchain.chains import RunnableChain

openai_api_key = os.getenv("OPENAI_API_KEY")

# LLMの設定
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                  openai_api_key=openai_api_key)

# サンプルデータを用意
sample_data = {
    "columns": ["Month", "Sales", "Profit"],
    "data": [
        ["January", 10000, 2000],
        ["February", 15000, 3000],
        ["March", 12000, 2500],
    ]
}

# グラフを生成する関数
def generate_dynamic_graph(data, graph_config):
    # DataFrameに変換
    df = pd.DataFrame(data["data"], columns=data["columns"])
    
    # グラフ設定の読み取り
    graph_type = graph_config.get("graph_type", "line")
    x_axis = graph_config.get("x_axis", "index")
    y_axes = graph_config.get("y_axes", [])
    title = graph_config.get("title", "Dynamic Graph")
    xlabel = graph_config.get("xlabel", x_axis)
    ylabel = graph_config.get("ylabel", "Values")
    
    # プロット
    fig, ax = plt.subplots()
    
    if graph_type == "bar":
        df.plot(x=x_axis, y=y_axes, kind="bar", ax=ax)
    elif graph_type == "line":
        df.plot(x=x_axis, y=y_axes, kind="line", ax=ax)
    elif graph_type == "scatter" and len(y_axes) == 2:
        df.plot.scatter(x=y_axes[0], y=y_axes[1], ax=ax)
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")
    
    # タイトルと軸ラベル
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    # 画像を保存
    plt.savefig("dynamic_graph_output.png")
    print("Graph saved as 'dynamic_graph_output.png'")

# LangGraphノードの定義
class DynamicGraphNode(RunnableNode):
    def invoke(self, state: State):
        # データと設定を取得
        data = state["data"]
        graph_config = state["graph_config"]
        
        # グラフ生成
        generate_dynamic_graph(data, graph_config)
        
        # 状態を更新
        state["message"] = "Dynamic graph generated and saved as 'dynamic_graph_output.png'"
        return state

# LangGraphのチェーンを構築
classify_prompt = PromptTemplate(
    input_variables=["data"],
    template=(
        "You are given the following dataset in JSON format:\n\n{data}\n\n"
        "Analyze the dataset and determine an appropriate graph type to visualize it. "
        "Provide the following configuration in JSON format:\n"
        "- `graph_type`: The type of graph (e.g., 'bar', 'line', 'scatter').\n"
        "- `x_axis`: The column to use for the X-axis.\n"
        "- `y_axes`: A list of columns to plot on the Y-axis.\n"
        "- `title`: The title of the graph.\n"
        "- `xlabel`: The label for the X-axis.\n"
        "- `ylabel`: The label for the Y-axis."
    )
)

# LangGraphチェーンの構成
chain = (
    classify_prompt
    | llm
    | DynamicGraphNode()
)

# 実行
state = State(data=sample_data)
response = chain.invoke(state)

# 結果表示
print(response["message"])

