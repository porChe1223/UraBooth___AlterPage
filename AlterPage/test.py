from typing import Annotated, Literal
from typing_extensions import Sequence,TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import FunctionMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import openai
from langchain_experimental.utilities import PythonREPL
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain.schema import Document
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import time
import random
import re
import functools
import getpass
import json
import matplotlib.pyplot as plt
import langchain.vectorstores
import logging
import chainlit as cl
import requests

tavily_tool = TavilySearchResults(max_results=5)
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# 環境変数読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

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

""""""""""""""""""""""""""""""
############ ブログの改善提案 ############

#分類器LLMの出力を固定するクラスを指定する
class MessageType(BaseModel):
    message_type: str = Field(description="The type of the message", example="search")
#AIエージェントが選択するツールの選択出力を固定するクラスを指定する
class ToolType(BaseModel):
    message_type: str = Field(description="Must use the Tool", example="tool01" or "tool02")

#AIエージェントがユーザの質問（7個の質問かその他の質問か）を分類し、適切なワークフロー（tool01かtool2）を呼び出すための関数の定義

#出力を固定するため
tools = llm.with_structured_output(ToolType)

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


# tool01で利用する、ユーザの質問を分類するLLMノードを定義

#出力を固定するため
classifier = llm.with_structured_output(MessageType)

#tool01に分類されたノードで実行される関数を定義,さらに取得するデータごとに分類する

def specific_analyze(llm, user_prompt):
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"], 
        template = txt_read("resource/sys_prompt/specific_analyze.txt") + "\n\nプロンプト: {user_prompt}" ,#7こに細かく分類させるプロンンプと
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})
    
#tool01ワークフローの質問回答LLMノードを定義

# tool1の最終出力ノードを定義
#def response(State):
    #return State

#tool02に分類されたノードで実行される関数を定義

###############
# Agentの設定 #
###############

def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"

repl = PythonREPL()

@tool
def python_repl(code: Annotated[str, "チャートを生成するために実行する Python コード"]):
    """
    これを使用して Python コードを実行します。
    値の出力を確認したい場合は、`print(...)` で出力する必要があります。
    これはユーザーに表示されます。
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\\\\n`python\\\\\\\\n{code}\\\\\\\\n`\\\\nStdout: {result}"

tools = [tavily_tool, python_repl]
tool_executor = ToolExecutor(tools)

def create_agent(llm, tools, system_message: str):
    """エージェントを作成します。"""
    functions = [convert_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " あなたは他のアシスタントと協力して、役に立つ AI アシスタントです。 "
                " 提供されたツールを使用して、質問の回答に進みます。 "
                " 完全に答えることができなくても大丈夫です。 "
                " 別のツールを備えた別のアシスタントが中断したところからサポートします。進歩するためにできることを実行してください。"
                " あなたまたは他のアシスタントが最終的な回答または成果物を持っている場合は、チームが停止することがわかるように、回答の前に「FINAL ANSWER」を付けます。"
                " 次のツールにアクセスできます: {tool_names}.\\\\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

research_agent= create_agent(
    llm,
    [tavily_tool],
    system_message="Chart Generatorが使用する正確なデータを提供する必要があります。",
)

chart_agent= create_agent(
    llm,
    [python_repl],
    system_message="表示したグラフはすべてユーザーに表示されます。",
)

############################
# 分岐1のワークフローを作成する #
############################
#子ノードを流れるデータの型を定義

#ノードの宣言
# 開始ノード


############################
# 分岐2のワークフローを作成する #
############################

#各ワークフローは整理のため、子ノードとして定義する

#子ノードを流れるStateを定義
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

tool_executor = ToolExecutor(tools)

def create_agent(llm, tools, system_message: str):
    """エージェントを作成します。"""
    functions = [convert_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " あなたは他のアシスタントと協力して、役に立つ AI アシスタントです。 "
                " 提供されたツールを使用して、質問の回答に進みます。 "
                " 完全に答えることができなくても大丈夫です。 "
                " 別のツールを備えた別のアシスタントが中断したところからサポートします。進歩するためにできることを実行してください。"
                " あなたまたは他のアシスタントが最終的な回答または成果物を持っている場合は、チームが停止することがわかるように、回答の前に「FINAL ANSWER」を付けます。"
                " 次のツールにアクセスできます: {tool_names}.\\\\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

research_agent= create_agent(
    llm,
    [tavily_tool],
    system_message="Chart Generatorが使用する正確なデータを提供する必要があります。",
)

chart_agent= create_agent(
    llm,
    [python_repl],
    system_message="表示したグラフはすべてユーザーに表示されます。",
)

#ノードの宣言

def node_tool(state):
    """
    これにより、グラフ内のツールが実行されます。
    エージェントのアクションを受け取り、そのツールを呼び出して結果を返します。
    """
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

def node_agent(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, FunctionMessage):
        pass
    else:
        valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)  # 無効な文字を置換
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=valid_name)
    return {
        "messages": [result],
        "sender": name,
    }


# グラフの作成
workflow2= StateGraph(AgentState)

# ノードの作成
node_researcher= functools.partial(node_agent, agent=research_agent, name="node_researcher")
node_chart= functools.partial(node_agent, agent=chart_agent, name="node_chart")

workflow2.add_node("node_researcher", node_researcher)
workflow2.add_node("node_chart", node_chart)
workflow2.add_node("call_tool", node_tool)



#スタートノードの設定
workflow2.set_entry_point("node_researcher")

# エッジの作成
workflow2.add_conditional_edges(
    "node_researcher", # <- 起点のエージェント
    router, # <- ルーターの戻り値を条件とするという定義
    {"continue": "node_chart", "call_tool": "call_tool", "end": END}, # <- 3パターンの定義
)

workflow2.add_conditional_edges(
    "node_chart",
    router,
    {"continue": "node_researcher", "call_tool": "call_tool", "end": END},
)

workflow2.add_conditional_edges(
    "call_tool",
		lambda x: x["sender"],
    {
        "node_researcher": "node_researcher",
        "node_chart": "node_chart",
    },
)

########################################
""""""""""""""""""""""""""""""""""""
# データ取得の関数定義

urls1 = ["https://cosmosdbdatagetter.azurewebsites.net/data",]
urls2 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報",]
urls3 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=トラフィックソース関連情報",]
urls4 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=ユーザー行動関連情報",]
urls5 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=サイト内検索関連情報",]
urls6 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=デバイスおよびユーザ属性関連情報",]
urls7 = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=時間帯関連情報",]

def call_Analyze(llm, user_prompt):
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=openai_api_key
                    )
    url = "https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報"
    print(url)
    #url = "https://～.azurewebsites.net/data?group=ページ関連情報"
    response = requests.get(url)
    print(response.text)

    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template =  response.text,
        template_format="jinja2"
        )
    # チェーンの宣言
    chain = (
        system_prompt_Evaluate
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#AIエージェントがユーザの質問（7個の質問かその他の質問か）を分類し、適切なワークフロー（tool01かtool2）を呼び出すための関数の定義

def select_tool(llm, user_prompt):
    # プロンプトの作成
    classification_prompt = PromptTemplate(
        input_variables = ["user_prompt"], #tool1かtool2が選択される
        template = txt_read("resource/sys_prompt/classification_prompt.txt") + "\n\nプロンプト: {user_prompt}"
    )
    # チェーンの宣言
    chain = (
        classification_prompt
        | llm
    )
    # チェーンの実行
    return chain.invoke({"user_prompt": user_prompt})

#############################################

# プロンプトの評価
def call_Review(llm, user_prompt):
    # プロンプト評価用のシステムプロンプト
    system_prompt_Evaluate = PromptTemplate(
        input_variables = ["user_prompt"],
        template = txt_read("resource/sys_prompt/evaluation.txt") + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
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
        template = txt_read("resource/rag_data/alterbooth.txt") + "\n\nプロンプト: {user_prompt}"
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

### ここにデータ取得の関数を追加する ###

# 全データ取得ノード
def node_DataGet(state: State, config: RunnableConfig, url:list=urls1):
    if url is None:
        url = ["https://cosmosdbdatagetter.azurewebsites.net/data"]
    prompt = state["message"]
    result_data = call_Analyze(llm,prompt)
    return {"message":result_data}

# ページ関連情報取得ノード
def node_DataGet_page(state: State, config: RunnableConfig, url:list=urls2):

    url = ["https://cosmosdbdatagetter.azurewebsites.net/data?group=ページ関連情報"]

    prompt = state["message"]
    print(prompt)
    result_data = call_Analyze(llm, prompt)
    print(result_data)
    return {"message":result_data}

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
graph_builder.add_node("node_DataGet", node_DataGet)
graph_builder.add_node("node_DataGet_page", node_DataGet_page)
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

graph_builder.add_conditional_edges(
    "node_Analyze",
    lambda state: state["message_type"],
    {
        "tool01": "node_DataGet_page",
        "tool02": "node_DataGet",
    },
)

# Nodeをedgeに追加
graph_builder.add_edge("node_DataGet_page", "node_End")
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
