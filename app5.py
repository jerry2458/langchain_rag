import streamlit as st
from rag_functions5 import generate_detailed_explanation
from langchain.chat_models import AzureChatOpenAI
import os

# ✅ Streamlit 설정
st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 문제, 해설, 정답을 입력하면 AI가 친절한 해설을 생성해줍니다.")

# ✅ 모델별 설정값 정의 (각 환경 변수에서 직접 가져오기)
model_options = {
    "GPT-4": {
        "deployment_name": os.getenv("AZURE_GPT4_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_GPT4_API_VERSION"),
        "api_base": os.getenv("AZURE_GPT4_ENDPOINT"),
        "api_key": os.getenv("AZURE_GPT4_API_KEY"),
        "supports_temperature": True  # ✅ GPT-4는 temperature 지원
    },
    "GPT-o3-mini": {
        "deployment_name": os.getenv("AZURE_GPTo3_DEPLOYMENT_NAME"),  # ✅ 환경 변수명 수정 (오타 확인 필요)
        "api_version": os.getenv("AZURE_GPTo3_API_VERSION"),
        "api_base": os.getenv("AZURE_GPTo3_ENDPOINT"),
        "api_key": os.getenv("AZURE_GPTo3_API_KEY"),
        "supports_temperature": False  # ✅ GPT-o3-mini는 temperature 미지원
    }
}

# ✅ 모델 선택 UI
st.sidebar.header("⚙️ 모델 설정")
selected_model = st.sidebar.radio("모델 선택", list(model_options.keys()))

# ✅ 선택한 모델의 설정값 가져오기
selected_settings = model_options[selected_model]

# ✅ 사용자 입력창 생성 (문제, 해설, 정답 입력)
st.header("📝 문제 입력")
question_input = st.text_area("문제 입력", "이곳에 문제를 입력하세요.")
solution_input = st.text_area("기존 해설 입력", "이곳에 기존 해설을 입력하세요.")
answer_input = st.text_input("정답 입력", "이곳에 정답을 입력하세요.")

# ✅ 사용자 프롬프트 수정 가능
st.sidebar.header("📝 프롬프트 설정")
default_prompt = (
    "문제, 해설, 정답을 기반으로 새로운 해설을 작성하세요.\n"
    "설명은 초등학생도 이해할 수 있도록 친절하고 자세하게 작성해주세요.\n"
    "수식은 LaTeX 형식으로 유지하고, HTML 포맷을 사용해 가독성을 높여주세요.\n"
)
user_prompt = st.sidebar.text_area("프롬프트 수정", default_prompt, height=150)

# ✅ LLM 모델 설정 (GPT-o3-mini는 temperature=None을 명시적으로 설정)
if selected_settings["supports_temperature"]:
    llm = AzureChatOpenAI(
        deployment_name=selected_settings["deployment_name"],
        openai_api_version=selected_settings["api_version"],
        openai_api_base=selected_settings["api_base"],
        openai_api_key=selected_settings["api_key"],
        temperature=0.5  # ✅ GPT-4는 temperature 사용 가능
    )
else:
    llm = AzureChatOpenAI(
        deployment_name=selected_settings["deployment_name"],
        openai_api_version=selected_settings["api_version"],
        openai_api_base=selected_settings["api_base"],
        openai_api_key=selected_settings["api_key"]
    )

# ✅ 변환 실행 버튼
if st.button("🔄 해설 변환 실행"):
    with st.spinner("🔍 AI가 친절한 해설을 생성 중..."):
        transformed_solution = generate_detailed_explanation(
            llm, question_input, solution_input, answer_input, user_prompt
        )

    # ✅ 결과 출력
    st.header("✨ 변환된 친절한 해설")
    st.markdown(transformed_solution, unsafe_allow_html=True)
