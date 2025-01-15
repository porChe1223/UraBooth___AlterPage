from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
import openai
import getpass
import os
import json
import functools
import operator
import re
import langchain.vectorstores
import langchain.embeddings
import langchain_community.document_loaders
import langchain.text_splitter


openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", 
                 temperature=0, 
                 openai_api_key=openai_api_key
                 )


# CosmosDBデータ取得ツールを定義
#from FUNC_datagetter import data_get
#data_get_tool = data_get()
def data_get(llm, prompt):
    documents_url = ["https://cosmosdbdatagetter.azurewebsites.net/data?data_range=2024-9-28 to 2025-1-1",]
    loader = langchain_community.document_loaders.SeleniumURLLoader(urls=documents_url)  
    documents = loader.load() 

    # 読込した内容を分割する
    text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    #if "FINAL ANSWER" in last_message.content:
    if "FINAL ANSWER" in last_message.content or state.get("complete", False):
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

@tool
def analyze_tool(analysis: Annotated[str, "分析結果を生成するためのコード"]):
    """
    これを使用してデータを分析します。
    """
    return f"Analysis of data: {analysis}"

@tool
def data_select_tool(selected: Annotated[str, "データを選択するためのコード"]):
    """
    これを使用してデータを選択します。
    """
    return f"Selected data: {selected}"

tools = [data_select_tool, python_repl,analyze_tool ]

tool_executor = ToolExecutor(tools)

###############
# Stateの宣言 #
###############
class State(TypedDict):
    message: str
"""
def data_node(state: AgentState, config: RunnableConfig):
    prompt = state["message"]
    response = data_get(llm, prompt)
    # ここに他の処理を追加
    return { "message": response.content }
"""
def data_node(state, config):
    # デバッグログを追加して state の中身を確認
    print("DEBUG: state before processing:", state)
    
    # "message" が存在しない場合の対策
    if "message" not in state:
        raise KeyError("Expected 'message' in state but not found.")
    
    prompt = state["message"]
    print("DEBUG: Using prompt:", prompt)
    data = data_get(llm, prompt)
    response = data_get(llm, prompt)
    print("DEBUG: Retrieved documents:", data)
    # ここに他の処理を追加
    #return { "message": response }
    
    


def tool_node(state):
    """
    これにより、グラフ内のツールが実行されます。
    エージェントのアクションを受け取り、そのツールを呼び出して結果を返します。
    """
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    #tool_input = json.loads(
      #  last_message.additional_kwargs["function_call"]["arguments"]
    #)
    try:
        tool_input = json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        )
    except (KeyError, json.JSONDecodeError):
        tool_input = {}
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}  # Ensure 'messages' is used here

def create_agent(llm, tools, system_message: str):
    """エージェントを作成します。"""
    functions = [convert_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        {"messages": [system_message]}
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

selection_agent= create_agent(
    llm,
    [data_select_tool],
    system_message="データを選択するために使用する正確なデータを提供する必要があります。",
)

analyze_agent= create_agent(
    llm,
    [analyze_tool],
    system_message="Chart Generatorが使用する正確なデータを提供する必要があります。",
)

chart_agent= create_agent(
    llm,
    [python_repl],
    system_message="表示したグラフはすべてユーザーに表示されます。",
)

def agent_node(state, agent, name):
    prompt = state.get("message", "")
    data = state.get("data", "")
    combined_prompt = f"{prompt}\n\n---\n\n{data}"

    # エージェントを呼び出して結果を取得
    result = agent.invoke({"message": combined_prompt})
    print(f"DEBUG [{name}]: LLM result: {result}")

    if isinstance(result, FunctionMessage):
        pass
    else:
        valid_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)  # 無効な文字を置換
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=valid_name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


selection_node= functools.partial(agent_node, agent=selection_agent, name="Selector")
analysis_node= functools.partial(agent_node, agent=analyze_agent, name="Analyzer")
chart_node= functools.partial(agent_node, agent=chart_agent, name="Chart Generator")


workflow = StateGraph(State)

workflow.add_node("data_node", data_node)
workflow.add_node("Selector", selection_node)
workflow.add_node("Analyzer", analysis_node)
workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("data_node")

workflow.add_edge("data_node", "Selector")
workflow.add_conditional_edges(
    "Selector",
    router,
    {"continue": "Chart Generator", "call_tool": "call_tool","end": END},
)
workflow.add_conditional_edges(
    "Chart Generator", # <- 起点のエージェント
    router, # <- ルーターの戻り値を条件とするという定義
    {"continue": "Analyzer", "call_tool": "call_tool", "end": END},# <- 3パターンの定義
)
workflow.add_conditional_edges(
    "Analyzer",
    router,
    {"continue": "Chart Generator", "call_tool": "call_tool", "end": END},
)
workflow.add_conditional_edges(
    "call_tool",
		lambda x: x["sender"],
    {
        "Selector": "Selector",
        "Analyzer": "Analyzer",
        "Chart Generator": "Chart Generator",
    },
)

#END = "some_value"

graph= workflow.compile()

# Graphの実行(引数にはStateの初期値を渡す)
initial_state = {
    "message": '{pageTitle}を教えて',
    "messages": []  # 'messages' キーを追加
}
graph.invoke(initial_state, debug=True)

