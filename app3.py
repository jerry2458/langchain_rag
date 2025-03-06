import streamlit as st
from rag_functions3 import load_html_explanation_data, load_and_split_pdf, create_vector_store, create_rag_chain

# ✅ 파일 경로 설정
csv_path = "./qbank_quest_danbi.csv"   # 문제 & 해설 데이터
pdf_path = "./em_5_2_5_c_e.pdf"       # 수학 단원 PDF 파일

# ✅ 데이터 로드 및 벡터 저장소 생성
st.sidebar.header("📂 데이터 로딩 중...")
explanations = load_html_explanation_data(csv_path)
pdf_texts = load_and_split_pdf(pdf_path)

explanation_store, pdf_store = create_vector_store(explanations, pdf_texts)
llm_chain, retriever, pdf_retriever = create_rag_chain(explanation_store, pdf_store)

st.title("📘 AI 수학 해설 도우미")
st.write("학생의 질문을 입력하면, 관련된 해설과 PDF 자료를 찾아서 설명해 줍니다!")

# ✅ 사용자 입력 받기
query = st.text_input("❓ 학생의 질문을 입력하세요:")

if st.button("질문하기"):
    if query:
        with st.spinner("🔍 GPT가 답변을 생성 중..."):
            # 🔹 문제 해설 검색
            explanation_docs = retriever.get_relevant_documents(query)
            explanation_text = "\n".join([doc.page_content for doc in explanation_docs])

            # 🔹 PDF 검색
            pdf_docs = pdf_retriever.get_relevant_documents(query)
            pdf_text = "\n".join([doc.page_content for doc in pdf_docs])

            # 🔹 GPT가 최종 답변 생성 (HTML 형식)
            response = llm_chain({
                "input_documents": explanation_docs + pdf_docs,
                "context": explanation_text,
                "pdf_context": pdf_text,
                "question": query
            })

            # ✅ 결과 출력
            st.markdown(response["output_text"], unsafe_allow_html=True)
    else:
        st.warning("❗ 질문을 입력해주세요.")
