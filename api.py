import getpass
import os

os.environ["TOGETHER_API_KEY"] = "api-key"

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="url",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="model-name",
)
