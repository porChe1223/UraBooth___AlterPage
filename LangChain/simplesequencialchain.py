from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["job"],
        template="{job}に一番オススメのプログラミング言語は何?\nプログラミング言語:"
    )
)
chat_prompt_template_1 = ChatPromptTemplate.from_messages([human_message_prompt_1])
chain_1 = LLMChain(llm=llm, prompt=chat_prompt_template_1)

human_message_prompt_2 = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["programming_language"],
            template="{programming_language}を学ぶためにやるべきことを3ステップで100文字で教えて。",
    )
)
chat_prompt_template_2 = ChatPromptTemplate.from_messages([human_message_prompt_2])
chain_2 = LLMChain(llm=llm, prompt=chat_prompt_template_2)

overall_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)
print(overall_chain("データサイエンティスト"))