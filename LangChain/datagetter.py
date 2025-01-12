from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import requests
# import os
# vitamin = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.5,openai_api_key="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")


# ツールの定義: データ取得用関数
def fetch_data_tool():
    def fetch_data(query: str) -> str:
        # 明示的にURLを指定
        url = "https://cosmosdbdatagetter.azurewebsites.net/data"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # データの最初の100文字のみを返す
        return response.text[:100] + "..."

    return Tool(
        name="FetchDataTool",
        func=fetch_data,
        description="Fetches data from the specified URL. Use this to get raw data.",
    )

# Agentの初期化
tools = [fetch_data_tool()]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Agentにタスクを依頼
# ここで明示的にURLを指定する必要はありません
query = "Fetch and explain the data from the predefined URL."
result = agent.run(query)

print(result)
