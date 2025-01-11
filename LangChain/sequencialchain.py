from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
import os
vitamin = os.getenv("OPENAI_API_KEY")

def txt_read(txtfile):
    f = open(txtfile, "r")
    data = f.read()
    f.close()
    return data

llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.5,openai_api_key=vitamin)
data1 = txt_read("aboutalldata.txt")
human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],                                        #ここ
            template=data1
    )
)
chat_prompt_template_1 = ChatPromptTemplate.from_messages([human_message_prompt_1])
chain_1 = LLMChain(llm=llm, prompt=chat_prompt_template_1, output_key="about_all_data")

data2 = txt_read("selectdata.txt")
human_message_prompt_2 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["about_all_data"],
            template=data2,
    )
)
chat_prompt_template_2 = ChatPromptTemplate.from_messages([human_message_prompt_2])
chain_2 = LLMChain(llm=llm, prompt=chat_prompt_template_2, output_key="select_data")

data3 = txt_read("analyze.txt")
human_message_prompt_3 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["select_data"],
            template=data3,
    )
)
chat_prompt_template_3 = ChatPromptTemplate.from_messages([human_message_prompt_3])
chain_3 = LLMChain(llm=llm, prompt=chat_prompt_template_3, output_key="analyze")

data4 = txt_read("advice.txt")
human_message_prompt_4 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["analyze"],
            template=data4,
    )
)
chat_prompt_template_4 = ChatPromptTemplate.from_messages([human_message_prompt_4])
chain_4 = LLMChain(llm=llm, prompt=chat_prompt_template_4, output_key="advice")

data5 = txt_read("answer.txt")
human_message_prompt_5 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["advice"],
            template=data5,
    )
)
chat_prompt_template_5 = ChatPromptTemplate.from_messages([human_message_prompt_5])
chain_5 = LLMChain(llm=llm, prompt=chat_prompt_template_5, output_key="answer")

overall_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4, chain_5],
    input_variables=["question"],
    output_variables=["select_data", "analyze", "advice", "answer"],
    verbose=True,
)
output = overall_chain.invoke({
    "question":"ブログの流入元について教えてください。(ここに質問を入力することになっています。)"
})
print(output)