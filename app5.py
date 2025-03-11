import streamlit as st
from rag_functions5 import generate_detailed_explanation
from langchain.chat_models import AzureChatOpenAI
import os

# ✅ Streamlit 설정
st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 문제, 해설, 정답을 입력하면 AI가 친절한 해설을 생성해줍니다.")

# ✅ 모델별 설정값 정의 (각 환경 변수에서 직접 가져오기)
model_options = {
    "GPT-4o": {
        "deployment_name": os.getenv("AZURE_GPT4o_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_GPT4o_API_VERSION"),
        "api_base": os.getenv("AZURE_GPT4o_ENDPOINT"),
        "api_key": os.getenv("AZURE_GPT4o_API_KEY"),
        "supports_temperature": True  # ✅ GPT-4는 temperature 지원
    },
    "GPT-4o-mini": {
        "deployment_name": os.getenv("AZURE_GPT4o_mini_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_GPT4o_mini_API_VERSION"),
        "api_base": os.getenv("AZURE_GPT4o_mini_ENDPOINT"),
        "api_key": os.getenv("AZURE_GPT4o_mini_API_KEY"),
        "supports_temperature": True  # ✅ GPT-4o-mini는 temperature 지원
    },
    "GPT-o3-mini": {
        "deployment_name": os.getenv("AZURE_GPTo3_mini_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_GPTo3_mini_API_VERSION"),
        "api_base": os.getenv("AZURE_GPTo3_mini_ENDPOINT"),
        "api_key": os.getenv("AZURE_GPTo3_mini_API_KEY"),
        "supports_temperature": False  # ✅ GPT-o3-mini는 temperature 미지원
    }
}

# ✅ 모델 선택 UI
st.sidebar.header("⚙️ 모델 설정")
selected_model = st.sidebar.radio("모델 선택", list(model_options.keys()))

# ✅ 선택한 모델의 설정값 가져오기
selected_settings = model_options[selected_model]

# ✅ LangChain 내부적으로 `temperature`가 포함되지 않도록 설정 확인
st.sidebar.write("🔍 LLM 설정값 확인")
st.sidebar.write(f"모델: {selected_model}")
st.sidebar.write(f"API Base: {selected_settings['api_base']}")
st.sidebar.write(f"API Version: {selected_settings['api_version']}")
st.sidebar.write(f"Supports Temperature: {selected_settings['supports_temperature']}")

# ✅ 사용자 프롬프트 수정 가능
st.sidebar.header("📝 프롬프트 설정")

default_prompt = ("이 데이터를 참고하여 초등학생도 이해할 수 있도록 쉽게 설명해주세요.\n"
                  "입력한 해설을 바탕으로 이해를 돕고, 기본 개념을 명확히 설명해주세요.\n"
                  "출력할 때 HTML 태그 구조를 반드시 유지하고, HTML 태그 텍스트가 그대로 화면에 보이도록 해주세요.")

user_prompt = st.sidebar.text_area("프롬프트 수정", default_prompt, height=200)


# ✅ 사용자 입력창 생성 (문제, 해설, 정답 입력)
st.header("📝 문제 입력")
question_input = st.text_area("문제 입력", "이곳에 문제를 입력하세요.")
solution_input = st.text_area("기존 해설 입력", "이곳에 해설을 입력하세요.")
answer_input = st.text_input("정답 입력", "이곳에 정답을 입력하세요.")

# ✅ LLM 모델 설정 (GPT-o3-mini는 temperature 인수를 완전히 제거)
if selected_settings["supports_temperature"]:
    llm = AzureChatOpenAI(
        deployment_name=selected_settings["deployment_name"],
        openai_api_version=selected_settings["api_version"],
        openai_api_base=selected_settings["api_base"],
        openai_api_key=selected_settings["api_key"],
        temperature=0  # ✅ GPT-4는 temperature 사용 가능
    )
else:
    # ✅ GPT-o3-mini의 경우 `temperature`를 아예 전달하지 않도록 설정
    llm = AzureChatOpenAI(
        deployment_name=selected_settings["deployment_name"],
        openai_api_version=selected_settings["api_version"],
        openai_api_base=selected_settings["api_base"],
        openai_api_key=selected_settings["api_key"],
        temperature=1
    )

# ✅ 변환 실행 버튼
if st.button("🔄 해설 변환 실행"):
    with st.spinner("🔍 AI가 친절한 해설을 생성 중..."):
        transformed_solution = generate_detailed_explanation(llm, question_input, solution_input, answer_input, user_prompt)

    # ✅ 결과 출력
    st.header("✨ 변환된 친절한 해설")
    st.markdown(transformed_solution, unsafe_allow_html= False)
