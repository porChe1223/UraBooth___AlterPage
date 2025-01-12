from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langgraph.prebuilt import ToolInvocation
import json

# モデルのセットアップ
tools = [TavilySearchResults(max_results=1, tavily_api_key="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")]
tool_executor = ToolExecutor(tools)

# トークンをストリーミングするために streaming=True を設定します
# これについての詳細はストリーミングセクションをご覧ください。
llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.5,openai_api_key="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"
,streaming=True)
#model = ChatOpenAI(temperature=0, streaming=True)
functions = [convert_to_openai_function(t) for t in tools]
model = llm.bind_functions(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 続行するかどうかを決定する関数を定義
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # 関数呼び出しがない場合は終了します
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # 関数呼び出しがある場合は続行します
    else:
        return "continue"


# モデルを呼び出す関数を定義
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # 既存のリストに追加されるので、リストを返します。
    return {"messages": [response]}


# ツールを実行する関数を定義
def call_tool(state):
    messages = state['messages']
    # 継続条件に基づき、最後のメッセージが関数呼び出しを含まれています
    last_message = messages[-1]
    # 関数呼び出しから ToolInvocation を構築します
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"]),
    )
    # tool_executorを呼び出し、レスポンスを返します
    response = tool_executor.invoke(action)
    # 応答を使って FunctionMessage を作成します
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # 既存のリストに追加されるので、リストを返します。
    return {"messages": [function_message]}

# 新しいグラフを定義
workflow = StateGraph(AgentState)

# 循環する二つのノードを定義
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# エントリーポイントとして `agent` を設定
# これはこのノードが最初に呼ばれることを意味します
workflow.set_entry_point("agent")

# 条件付きエッジを追加します
workflow.add_conditional_edges(
    # 最初に、開始ノードを定義します。`agent` を使用します。
    # これは `agent` ノードが呼び出された後に取られるエッジを意味します。
    "agent",
    # 次に、次に呼び出されるノードを決定する関数を渡します。
    should_continue,
    # 最後に、マッピングを渡します。
    # キーは文字列で、値は他のノードです。
    # END はグラフが終了することを示す特別なノードです。
    # `should_continue` を呼び出し、その出力がこのマッピングのキーに一致するものに基づいて、
    # 次に呼び出されるノードが決定されます。
    {
        # `continue` の場合は `action` ノードを呼び出します。
        "continue": "action",
        # それ以外の場合は終了します。
        "end": END
    }
)

# `action` から `agent` への通常のエッジを追加します。
# これは `action` が呼び出された後、次に `agent` ノードが呼ばれることを意味します。
workflow.add_edge('action', 'agent')

# 最後に、コンパイルします。
# これを LangChain Runnable にコンパイルし、他の runnable と同じように使用できるようにします
app = workflow.compile()

if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="東京の天気は？")]}
    print(app.invoke(inputs))

if __name__ == "__main__":
    app.get_graph().print_ascii()