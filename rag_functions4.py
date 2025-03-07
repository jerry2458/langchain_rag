import os
import re
import pandas as pd
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# ✅ (1) LaTeX 수식을 MathJax-friendly HTML로 변환
def convert_latex_to_mathjax(text):
    latex_regex = re.compile(r'\\\((.*?)\\\)')  # \( ... \) 형태 감지

    def replace_latex(match):
        latex_code = match.group(1)
        return f'<span class="mathjax">\\({latex_code}\\)</span>'

    return latex_regex.sub(replace_latex, str(text))


# ✅ (2) HTML 형식의 문제 및 해설 데이터 로드 (LaTeX 변환 적용)
def load_html_explanation_data(file_path):
    df = pd.read_csv(file_path)  # CSV에서 HTML 형식 데이터 불러오기
    explanations = []
    for _, row in df.iterrows():
        explanations.append({
            "question": convert_latex_to_mathjax(row["문제"]),
            "explanation": convert_latex_to_mathjax(row["해설"])
        })
    return explanations


# ✅ (3) PDF 파일 로드 및 텍스트 분할
def load_and_split_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs


# ✅ (4) 벡터 저장소 생성 (문제 해설 + PDF 단원 저장)
def create_vector_store(explanations, pdf_texts, persist_directory="chroma_db"):
    embeddings = OpenAIEmbeddings()

    # HTML 문제 해설을 ChromaDB에 추가
    explanation_texts = [f"<b>문제:</b> {ex['question']}<br><b>해설:</b> {ex['explanation']}" for ex in explanations]
    explanation_store = Chroma.from_texts(explanation_texts, embeddings,
                                          persist_directory=persist_directory + "/explanations")

    # PDF 단원도 ChromaDB에 추가
    pdf_store = Chroma.from_documents(pdf_texts, embeddings, persist_directory=persist_directory + "/pdfs")

    return explanation_store, pdf_store


# ✅ (5) RAG 기반 질의응답 체인 생성
def create_rag_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    prompt_template = PromptTemplate(
        template=(
            "학생의 질문과 기존 해설을 참고하여, 더 친절한 해설을 제공합니다 (HTML 형식 출력):\n\n"
            "🔹 <b>문제</b><br>{question}<br>\n"
            "🔹 <b>기존 해설</b><br>{context}<br>\n"
            "🔹 <b>새로운 해설</b><br>\n"
            "이전 해설보다 더욱 친절하고 이해하기 쉬운 방식으로 설명해 주세요.\n"
            "HTML과 LaTeX 수식을 유지해 주세요."
        ),
        input_variables=["context", "question"]
    )

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

