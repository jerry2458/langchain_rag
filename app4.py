import streamlit as st
from rag_functions4 import load_html_explanation_data, refine_question, refine_explanation
from langchain.chat_models import AzureChatOpenAI
import os
import streamlit.components.v1 as components

os.environ["openai_api_base"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

# ✅ 파일 경로 설정
csv_path = "./question20.csv"

# ✅ 데이터 로드
st.sidebar.header("📂 데이터 로딩 중...")
problems = load_html_explanation_data(csv_path)

# ✅ GPT 모델 설정
llm = AzureChatOpenAI(
    deployment_name="cats-aieng-prod-gpt4o-2024-05-13",
    openai_api_version="2024-05-01-preview",
    temperature=0.5
)

# ✅ MathJax 스크립트 추가 (LaTeX 수식 렌더링)
mathjax_script = """
<script>
MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] },
    svg: { fontCache: 'global' }
};
</script>
<script type="text/javascript" async
  src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""
st.components.v1.html(mathjax_script, height=0)

st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 GPT가 변환한 문제와 친절한 해설을 한 개씩 출력합니다.")

# ✅ 문제 & 해설 변환 수행 (각 문제마다 한 번씩 GPT 실행)
for index, problem in enumerate(problems):
    with st.spinner(f"🔍 GPT가 문제 {index+1} 변환 중..."):
        transformed_question = refine_question(llm, problem["question"])

    # ✅ 변환된 문제 & 해설 출력
    st.markdown(f"### 📝 문제 {index+1} (ID: {problem['question_id']})")  # ✅ 문항아이디 포함
    st.markdown(transformed_question, unsafe_allow_html=True)  # ✅ 변환된 문제 출력
    
    with st.spinner(f"🔍 GPT가 문제 {index+1} 해설을 변환 중..."):
        transformed_explanation = refine_explanation(llm, problem["explanation"])
    
    st.markdown("#### ✨ 새롭게 친절해진 해설")
    st.markdown(transformed_explanation, unsafe_allow_html=True)  # ✅ 변환된 해설 출력
