import streamlit as st
from rag_functions4 import load_html_explanation_data, generate_detailed_explanation
from langchain.chat_models import AzureChatOpenAI
import os

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
    temperature=1.0
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
st.write("📢 모든 문제와 친절한 해설을 한 페이지에서 확인하세요!")

# ✅ 문제 & 해설 변환 수행 (한 번만 GPT 실행)
converted_problems = []
with st.spinner("🔍 GPT가 모든 문제와 해설을 변환하는 중..."):
    for problem in problems:
        transformed_question = generate_detailed_explanation(llm, problem["question"], problem["explanation"])
        converted_problems.append({
            "question_id": problem["question_id"],
            "question": transformed_question  # ✅ GPT가 변환한 문제 사용
        })

# ✅ 변환된 문제 출력
for index, problem in enumerate(converted_problems):
    st.markdown(f"### 📝 문제 {index+1} (ID: {problem['question_id']})")  # ✅ 문항아이디 포함
    st.markdown(problem["question"], unsafe_allow_html=True)  # ✅ 변환된 문제 출력
