import getpass
import os
import json
import functools
import operator
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from typing import Annotated
from typing_extensions import Sequence,TypedDict
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
import re

tavily_tool = TavilySearchResults(max_results=5)

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# 環境変数読み込み
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

# Stateの設定
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

def agent_node(state, agent, name):
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
workflow= StateGraph(AgentState)


# ノードの作成
research_node= functools.partial(agent_node, agent=research_agent, name="Researcher")
chart_node= functools.partial(agent_node, agent=chart_agent, name="Chart Generator")

workflow.add_node("Researcher", research_node)
workflow.add_node("Chart Generator", chart_node)
workflow.add_node("call_tool", tool_node)



#スタートノードの設定
workflow.set_entry_point("Researcher")

# エッジの作成
workflow.add_conditional_edges(
    "Researcher", # <- 起点のエージェント
    router, # <- ルーターの戻り値を条件とするという定義
    {"continue": "Chart Generator", "call_tool": "call_tool", "end": END}, # <- 3パターンの定義
)

workflow.add_conditional_edges(
    "Chart Generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "end": END},
)

workflow.add_conditional_edges(
    "call_tool",
		lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "Chart Generator": "Chart Generator",
    },
)

graph= workflow.compile()

initial_state = {
    "messages": [
        HumanMessage(
            content="過去5年の日本のGDPを調査してください。次に調査した結果に基づいて質問に回答してください。",
            name="Researcher"  # 有効な名前を設定
        )
    ],
    "sender": "initial_sender"
}

for s in graph.stream(initial_state, debug=True):
    print(s)
    print("----")



