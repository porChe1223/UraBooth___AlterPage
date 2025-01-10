from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["adjective", "job"],
            template="{adjective}{job}に一番オススメのプログラミング言語は?\nプログラミング言語:",
    )
)
chat_prompt_template_1 = ChatPromptTemplate.from_messages([human_message_prompt_1])
chain_1 = LLMChain(llm=llm, prompt=chat_prompt_template_1, output_key="programming_language")

human_message_prompt_2 = HumanMessagePromptTemplate(
    prompt= PromptTemplate(
        input_variables=["programming_language"],
            template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
    )
)
chat_prompt_template_2 = ChatPromptTemplate.from_messages([human_message_prompt_2])
chain_2 = LLMChain(llm=llm, prompt=chat_prompt_template_2, output_key="learning_step")

overall_chain = SequentialChain(
    chains=[chain_1, chain_2],
    input_variables=["adjective", "job"],
    output_variables=["programming_language", "learning_step"],
    verbose=True,
)
output = overall_chain({
    "adjective": "ベテランの",
    "job": "データサイエンティスト",
})
print(output)