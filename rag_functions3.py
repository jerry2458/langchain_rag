import os
import pandas as pd
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ✅ (1) HTML 형식의 문제 및 해설 데이터 로드
def load_html_explanation_data(file_path):
    df = pd.read_csv(file_path)  # CSV에서 HTML 형식 데이터 불러오기
    explanations = []
    for _, row in df.iterrows():
        explanations.append({
            "question": row["문제"],
            "explanation": row["해설"]
        })
    return explanations


# ✅ (2) PDF 파일 로드 및 텍스트 분할
def load_and_split_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


# ✅ (3) 벡터 저장소 생성 (문제 해설 + PDF 단원 저장)
def create_vector_store(explanations, pdf_texts, persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()

    # HTML 문제 해설을 ChromaDB에 추가
    explanation_texts = [f"<b>문제:</b> {ex['question']}<br><b>해설:</b> {ex['explanation']}" for ex in explanations]
    explanation_store = Chroma.from_texts(explanation_texts, embeddings, persist_directory=persist_directory + "/explanations")

    # PDF 단원도 ChromaDB에 추가
    pdf_store = Chroma.from_documents(pdf_texts, embeddings, persist_directory=persist_directory + "/pdfs")

    return explanation_store, pdf_store


# ✅ (4) RAG 기반 질의응답 체인 생성
def create_rag_chain(explanation_store, pdf_store):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    retriever = explanation_store.as_retriever()  # 문제 해설 검색
    pdf_retriever = pdf_store.as_retriever()  # PDF 검색

    prompt_template = PromptTemplate(
        template=(
            "학생의 질문에 대한 해설과 관련 개념을 제공합니다 (HTML 형식 출력):\n\n"
            "🔹 <b>문제 해설</b><br>{context}<br>\n"
            "🔹 <b>관련 개념 PDF 단원</b><br>{pdf_context}<br>\n"
            "질문: {question}\n"
            "LaTeX 수식이 포함된 HTML 형식으로 답변을 제공하세요."
        ),
        input_variables=["context", "pdf_context", "question"]
    )

    llm_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

    return llm_chain, retriever, pdf_retriever
