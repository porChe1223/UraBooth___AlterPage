##########################################################################
################          マルチエージェントの設定        ####################
##########################################################################

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import tool
from langchain_core.messages import FunctionMessage
import getpass
import json
import functools
import re

tavily_tool = TavilySearchResults(max_results=5)

# 環境変数読み込み
tavily_api_key = os.getenv("TAVILY_API_KEY")


# エージェントのルートを決める関数
def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "end"
    return "continue"

repl = PythonREPL()

@tool

# チャートを生成するために実行する Python コード
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


#ワークフローは整理のため、子ノードとして定義する

#子ノードを流れるStateを定義
class AgentState(TypedDict):
    messages: str
    sender: str

tool_executor = ToolExecutor(tools)

# エージェントの作成
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

# web検索エージェント
research_agent= create_agent(
    llm,
    [tavily_tool],
    system_message="Chart Generatorが使用する正確なデータを提供する必要があります。",
)

# チャートエージェント
chart_agent= create_agent(
    llm,
    [python_repl],
    system_message="表示したグラフはすべてユーザーに表示されます。",
)

# エージェントノードの出力を集約する関数
def collect_results(state):
    return {"final_output": state["messages"]}

#ノードの宣言

# ツールノード
def node_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    return {"messages": [function_message]}

# エージェントノード
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
workflow2.add_node("collect_results", collect_results)


#スタートノードの設定
workflow2.set_entry_point("node_researcher")

# エッジの作成
workflow2.add_conditional_edges(
    "node_researcher", # <- 起点のエージェント
    router, # <- ルーターの戻り値を条件とするという定義
    {"continue": "node_chart", "call_tool": "call_tool", "end": "collect_results"}, # <- 3パターンの定義
)

workflow2.add_conditional_edges(
    "node_chart",
    router,
    {"continue": "node_researcher", "call_tool": "call_tool", "end": "collect_results"},
)

workflow2.add_conditional_edges(
    "call_tool",
		lambda x: x["sender"],
    {
        "node_researcher": "node_researcher",
        "node_chart": "node_chart",
    },
)

workflow2.set_finish_point("collect_results")

workflow2 = workflow2.compile()
########################################

graph_builder.add_node("workflow2", workflow2)

graph_builder.add_edge("node_DataGet", "workflow2")
graph_builder.add_edge("workflow2", "node_End")