import streamlit as st
from rag_functions4 import load_html_explanation_data, create_rag_chain

# ✅ 파일 경로 설정
csv_path = "qbank_quest_danbi.csv"

# ✅ 데이터 로드
st.sidebar.header("📂 데이터 로딩 중...")
problems = load_html_explanation_data(csv_path)

# ✅ GPT 모델 설정
llm_chain = create_rag_chain()

st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 모든 문제와 친절한 해설을 한 페이지에서 확인하세요!")

# ✅ 문제 & GPT 해설 출력
for index, problem in enumerate(problems):
    st.markdown(f"### 📝 문제 {index + 1}")
    st.markdown(problem["question"], unsafe_allow_html=True)  # HTML 문제 출력

    with st.spinner(f"🔍 GPT가 문제 {index + 1} 해설을 생성 중..."):
        response = llm_chain.run({
            "question": problem["question"],
            "explanation": problem["explanation"],
            "new_explanation": ""
        })

    st.markdown("#### ✨ 새롭게 친절해진 해설")
    st.markdown(response, unsafe_allow_html=True)  # GPT 변환 해설 출력
