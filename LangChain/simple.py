#from langchain.chat_models import ChatOpenAI
#from openai import OpenAI
from langchain.llms import OpenAI
# from langchain_google_vertexai import ChatVertexAI
# from langchain_community.chat_models import ChatGoogleGenerativeAI
# from google.cloud import aiplatform
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["job"],
				template="{job}に一番オススメのプログラミング言語は何?"
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chain = LLMChain(
    llm=OpenAI(model="gpt-4o-mini",openai_api_key="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"),
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.5,openai_api_key=OPENAI_API_KEY),
    prompt=chat_prompt_template
)

print(chain("データサイエンティスト"))