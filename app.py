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

# 1. PDF 텍스트 추출 (페이지 정보 포함)
def extract_text_with_page_info(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        documents = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # 텍스트가 존재하는 경우에만 처리
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={"source": os.path.basename(pdf_path), "page": page_num + 1}
                    )
                )
    return documents


# 2. 텍스트 분할 (메타데이터 유지)
def split_documents_with_metadata(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 분할 크기
        chunk_overlap=200  # 중복 영역
    )
    split_documents = []
    for doc in documents:
        splits = text_splitter.split_text(doc.page_content)
        for split in splits:
            split_documents.append(
                Document(
                    page_content=split,
                    metadata=doc.metadata  # 원본 메타데이터 유지
                )
            )
    return split_documents


# 3. 벡터 저장소 생성
def create_vector_store_with_metadata(documents, vectorstore_path):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=vectorstore_path)
    vectorstore.persist()  # 저장소를 로컬에 저장
    return vectorstore


# 4. PDF 폴더 처리
def process_pdf_folder_to_vectorstore(pdf_folder, vectorstore_path):
    documents = []

    # 모든 PDF 파일 처리
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")

            # 텍스트 추출 (페이지 정보 포함)
            page_documents = extract_text_with_page_info(pdf_path)
            documents.extend(page_documents)

    if not documents:
        raise ValueError("No valid documents found for vector store creation.")

    # 텍스트 분할 (메타데이터 유지)
    split_docs = split_documents_with_metadata(documents)

    # 벡터 저장소 생성
    print("Creating vector store...")
    vectorstore = create_vector_store_with_metadata(split_docs, vectorstore_path)
    print(f"Vector store saved at {vectorstore_path}.")
    return vectorstore


# 5. 벡터 저장소 로드
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
    자세하고 상세하게 하나의 문장을 완성하여 답변을 완성해주십시오.
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


# 7. 메인 실행
def main():
    # 벡터 저장소 경로
    vectorstore_path = "./chroma_db"  # 벡터 저장소 경로

    vectorstore = load_vectorstore(vectorstore_path)

    # QA 체인 생성
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vectorstore)

    # 질문 처리
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break

        result = qa_chain({"query": query})
        print("\n답변:")
        print(result["result"])

        print("\n출처:")
        for doc in result["source_documents"]:
            source = doc.metadata["source"]
            page = doc.metadata.get("page", "알 수 없음")  # 페이지 정보가 없으면 기본값
            print(f"- 파일: {source}, 페이지: {page}")


if __name__ == "__main__":
    main()
