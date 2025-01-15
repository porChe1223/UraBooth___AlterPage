from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
import 
import os
import time
import glob
import random
import matplotlib.pyplot as plt
import logging
import langchain.vectorstores
import langchain.embeddings
import langchain_community.document_loaders
import langchain.text_splitter
import json

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

########## ブログの改善提案 ##########

#分類器LLMの出力を固定するクラスを指定する
class MessageType(BaseModel):
    message_type: str = Field(description="The type of the message", example="search")
#AIエージェントが選択するツールの選択出力を固定するクラスを指定する
class ToolType(BaseModel):
    message_type: str = Field(description="Must use the Tool", example="tool01" or "tool02")

#AIエージェントがユーザの質問（7個の質問かその他の質問か）を分類し、適切なワークフロー（tool01かtool2）を呼び出すための関数の定義

#出力を固定するため
tools = llm.with_structured_output(ToolType)

def select_tool(State):
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        prompt = txt_read("RAG_classification_prompt.txt"),
        input_variables = ["messages"]
    )
    if State["messages"]:
        return {
            "message_type": tools.invoke(classification_prompt.format(user_message=State["messages"])).message_type,
            "messages": State["messages"]
            }
    else:
        return {"message": "No user input provided"}

# tool01からtool02までで利用する、ユーザの質問を分類するLLMノードを定義

#出力を固定するため
classifier = llm.with_structured_output(MessageType)

#tool01に分類されたノードで実行される関数を定義
def specific_analyze(State):
    # プロンプトの作成
    classification_prompt1 = PromptTemplate(
        prompt = txt_read("RAG_specific_analyze.txt"),
        input_variables = ["messages"]
    )
    if State["messages"]:
        return {
            "message_type": classifier.invoke(classification_prompt1.format(user_message=State["messages"])).message_type,
            "messages": State["messages"]
            }
    else:
        return {"message": "No user input provided"}
    

#tool02に分類されたノードで実行される関数を定義
def classify(State):
    # プロンプトの作成
    classification_prompt2 = PromptTemplate(
        prompt = txt_read("RAG_analyze_evaluate_suggest.txt"),
        input_variables = ["messages"]
    )
    if State["messages"]:
        return {
            "message_type": classifier.invoke(classification_prompt2.format(user_message=State["messages"])).message_type,
            "messages": State["messages"]
            }
    else:
        return {"message": "No user input provided"}
    

########################################


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

# データ取得の関数定義

def data_get():
    documents_url = ["https://cosmosdbdatagetter.azurewebsites.net/data?data_range=2024-9-28 to 2025-1-1",]


    loader = langchain_community.document_loaders.SeleniumURLLoader(urls=documents_url)  # 修正
    documents = loader.load() 

    # 読込した内容を分割する
    text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )
    docs = text_splitter.split_documents(documents)

    # OpenAIEmbeddings の初期化
    embedding = OpenAIEmbeddings()

    def get_embedding(text, model):
        text = text.replace("\n", " ")
        res = openai.embeddings.create(input = [text], model=model).data[0].embedding
        return res
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding
    )

###############
# Stateの宣言 #
###############
class State(TypedDict):
    data : str
    message: str
    message_type: str    #ツール名を格納

##############
# Nodeの宣言 #
##############
# 開始ノード
def node_Start(state: State, config: RunnableConfig):
    prompt = state["message"]  # LLMに渡すプロンプト
    return {"message": prompt} # 新しいStateに格納

# ブログの改善ノード
def node_Analyze(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = select_tool(llm, prompt)
    return {"message": f"分析: {prompt}"}

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

# 具体的な質問回答ノード
def node_SpecificAnalyze(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = specific_analyze(llm, prompt)
    return {"message": response.content}

# その他の分析、評価、提案ノード
def node_Classify(state: State, config: RunnableConfig):
    prompt = state["message"]
    response = classify(llm, prompt)
    return {"message": response.content}

# 全データ取得ノード
def node_DataGet(state: State, config: RunnableConfig):
    prompt = state["message"]
    data_get()
    return {"data": f"データ取得完了: {data_get.content}","message":prompt}



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
graph_builder.add_node("node_SpecificAnalyze", node_SpecificAnalyze)
graph_builder.add_node("node_Classify", node_Classify)
graph_builder.add_node("node_DataGet", node_DataGet)

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
# graph_builder.add_edge("node_Analyze", "node_A1")

# Graphの終点を宣言
graph_builder.set_finish_point("node_Analyze")
graph_builder.set_finish_point("node_Review")
graph_builder.set_finish_point("node_Others")

# Graphをコンパイル
graph = graph_builder.compile()

# Graphの実行(引数にはStateの初期値を渡す)
graph.invoke({'message': '「こんにちは」'}, debug=True)