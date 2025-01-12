from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
import os
vitamin = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=vitamin)

def txt_read(txtfile):
    f = open(txtfile, "r")
    data = f.read()
    f.close()
    return data

# システムプロンプトを読み込む
global_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
 # システムプロンプトに変数がない場合、空リストを指定
        template=txt_read("globalprompt.txt")  # 読み込んだテキストをテンプレートとして設定
    )
)

evaluation_system_prompt = txt_read("evaluationprompt.txt")

# プロンプト評価のためのテンプレート
evaluation_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template=evaluation_system_prompt + "\n\nプロンプト: {user_prompt}\n\nプロンプトの評価:"
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

data1 = txt_read("aboutalldata.txt")
human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],                                       
            template=data1
    )
)
chat_prompt_template_1 = ChatPromptTemplate.from_messages([global_system_prompt,human_message_prompt_1])
chain_1 = LLMChain(llm=llm, prompt=chat_prompt_template_1, output_key="about_all_data")

data2 = txt_read("selectdata.txt")
human_message_prompt_2 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["about_all_data"],
            template=data2,
    )
)
chat_prompt_template_2 = ChatPromptTemplate.from_messages([global_system_prompt,human_message_prompt_2])
chain_2 = LLMChain(llm=llm, prompt=chat_prompt_template_2, output_key="select_data")

data3 = txt_read("analyze.txt")
human_message_prompt_3 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["select_data"],
            template=data3,
    )
)
chat_prompt_template_3 = ChatPromptTemplate.from_messages([global_system_prompt,human_message_prompt_3])
chain_3 = LLMChain(llm=llm, prompt=chat_prompt_template_3, output_key="analyze")

data4 = txt_read("advice.txt")
human_message_prompt_4 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["analyze"],
            template=data4,
    )
)
chat_prompt_template_4 = ChatPromptTemplate.from_messages([global_system_prompt,human_message_prompt_4])
chain_4 = LLMChain(llm=llm, prompt=chat_prompt_template_4, output_key="advice")

data5 = txt_read("answer.txt")
human_message_prompt_5 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["advice"],
            template=data5,
    )
)
chat_prompt_template_5 = ChatPromptTemplate.from_messages([global_system_prompt,human_message_prompt_5])
chain_5 = LLMChain(llm=llm, prompt=chat_prompt_template_5, output_key="answer")

overall_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4, chain_5],
    input_variables=["user_input"],
    output_variables=["select_data", "analyze", "advice", "answer"],
    verbose=True,
)

def process_input(user_input: str):
    # 入力文に「評価」という単語が含まれているかを確認
    if "プロンプト" in user_input and "評価" in user_input:
        print("プロンプトを評価します。")
        # プロンプトの評価・改善を実行
        response = evaluation_chain.invoke({"user_prompt": user_input})
    else:
        # 通常の処理を実行
        response = overall_chain.invoke({"user_input": user_input})
    
    return response

if __name__ == "__main__":
    #user_input = "ブログの流入元について教えてください。(ここに質問を入力することになっています。)プロンプトを評価してください。"
    user_input = "私のブログは旅行に関するもので、ターゲットオーディエンスは20代から30代の若者です。ブログの流入元として、SEO、SNS、リファラルの効果を比較したいです。各流入元の特徴をリスト形式で説明し、具体的なデータや事例を含めてください。また、流入元を改善するための新しいアイデアも提案してください。プロンプトを評価してください。"
    output = process_input(user_input)
    print(output)
    print("評価に従ってプロンプトを改善し、再度質問してみてください!")