from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

#import os
#OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# テンプレート文章を定義し、プロンプトを作成
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは優秀な校正者です。"),
    ("user", "次の文章に誤字があれば訂正してください。\n{sentences_before_check}")
])

# OpenAIのモデルのインスタンスを作成
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) #引数でLLMの設定を行う

# OpenAIのAPIにこのプロンプトを送信するためのチェーンを作成
chain = prompt | llm | StrOutputParser()

# チェーンを実行し、結果を表示
print(chain.invoke({"sentences_before_check": "こんんんちわ、真純です。"}))