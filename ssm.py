import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.language_models.llms import LLM
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import requests

# 加载环境变量
os.environ["TOGETHER_API_KEY"] = "api-key"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")

# 定义豆包模型类
class DoubaoLLM(LLM):
    def __init__(self, api_key, model="model-name", base_url="https://api.together.xyz"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _call(self, prompt, stop=None):
        return self.generate(prompt)

    def _llm_type(self):
        return "doubao"

    # 初始化豆包模型


doubao_llm = DoubaoLLM(api_key=TOGETHER_API_KEY)

# 加载文档
loader = DirectoryLoader(path='./data')
documents = loader.load()

# 创建嵌入和向量存储
embeddings = OpenAIEmbeddings(model_name="gpt-3.5-turbo")  # 使用与 OpenAI 兼容的嵌入模型
vector_store = FAISS.from_documents(documents, embeddings)

# 创建检索器
retriever = vector_store.as_retriever()

# 创建 RAG 模型
rag_model = RetrievalQA(llm=doubao_llm, retriever=retriever)

# 进行查询
query = "介绍下如何计算主营业务毛利"
response = rag_model({"query": query})
print("结果response: ", response['result'])
