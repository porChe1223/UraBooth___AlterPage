from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
vitamin = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.5,openai_api_key=vitamin)
prompt = PromptTemplate(
    input_variables=["job"],
    template="{job}に一番オススメのプログラミング言語は何?"
)


# システムプロンプトを読み込む
def load_system_prompt(file_path):
    p = open(file_path, "r")    
    system_prompt = p.read()
    p.close()
    return system_prompt

# システムプロンプトファイルを読み込む
system_prompt = load_system_prompt("evaluationprompt.txt")

# プロンプト評価のためのテンプレート
evaluation_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template=system_prompt + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
)

# チェーンを定義
general_chain = LLMChain(llm=llm, prompt=prompt)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def process_input(user_input: str):
    # 入力文に「評価」という単語が含まれているかを確認
    if "プロンプト" in user_input and "評価" in user_input:
        print("プロンプトを評価します。")
        # プロンプトの評価・改善を実行
        response = evaluation_chain.run(user_prompt=user_input)
    else:
        # 通常の処理を実行
        response = general_chain.run(job=user_input)
    
    return response

# メイン処理
if __name__ == "__main__":
    # ユーザー入力（例）
    user_input = "データサイエンティストが直面する典型的なプロジェクトに基づいて、最も適したプログラミング言語を3つ挙げ、それぞれの言語の利点を具体的に説明してください。特に、言語の人気、学習の容易さ、コミュニティのサポートの観点から評価してください。プロンプトの評価を行なって"
    
    # 処理を実行
    output = process_input(user_input)
    print(output)
