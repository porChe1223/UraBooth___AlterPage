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
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("データサイエンティスト")
print(response)
