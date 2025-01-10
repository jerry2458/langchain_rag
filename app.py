import streamlit as st
from rag_functions import load_vectorstore, create_qa_chain

st.title("RAG Q&A 시스템")

# 사용자 입력
question = st.text_input("해당 주제에 대해 질문하세요:")

if question:
    if st.button("답변 받기"):
        with st.spinner("처리 중..."):

            print("Loading existing vector store...")
            vectorstore = load_vectorstore(vectorstore_path)

            # QA 체인 생성
            print("Creating QA chain...")
            qa_chain = create_qa_chain(vectorstore)

            # 질문에 대한 답변 생성
            result = qa_chain({"query": query})

            st.subheader("답변:")
            st.write(result["result"])

            st.subheader("출처:")
            for doc in result["source_documents"]:
                source = doc.metadata["source"]
                page = doc.metadata.get("page", "알 수 없음")  # 페이지 정보가 없으면 기본값
                st.write(f"- 파일: {source}, 페이지: {page}")
                st.write("---")

st.sidebar.title("소개")
st.sidebar.info(
    "이 앱은 RAG(검색 증강 생성) 시스템을 시연합니다. "
)
