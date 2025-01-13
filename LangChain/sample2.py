from langchain.chains import LLMChain
from langchain.chains.base import Chain

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + "\n" + output_2}

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt = PromptTemplate(
        input_variables=["job"],
            template="{job}に一番オススメのプログラミング言語は?\nプログラミング言語:",
    )
)
chat_prompt_template_1 = ChatPromptTemplate.from_messages([human_message_prompt_1])
chain_1 = LLMChain(llm=llm, prompt=chat_prompt_template_1)

human_message_prompt_1 = HumanMessagePromptTemplate(
    prompt = PromptTemplate(
        input_variables=["job"],
            template="{job}の平均年収は？\n平均年収:",
    )
)
chat_prompt_template_2 = ChatPromptTemplate.from_messages([human_message_prompt_1])
chain_2 = LLMChain(llm=llm, prompt=chat_prompt_template_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2, verbose=True)
print(concat_chain.run("データサイエンティスト"))