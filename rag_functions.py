import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import pdfplumber
import pysqlite3 as sqlite3

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 벡터 저장소 로드
def load_vectorstore(vectorstore_path):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
    return vectorstore


# 6. 질문-응답 체인 생성
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever()

    prompt_template = """아래의 문맥을 사용하여 질문에 답하십시오.
    만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
    검색을 통해 정확한 답변을 자세하게 설명해주는 선생님처럼 답해주십시오.
    {context}
    질문: {question}
    답변:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa_chain

