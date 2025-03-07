import streamlit as st
from rag_functions4 import load_html_explanation_data, generate_detailed_explanation
from langchain.chat_models import AzureChatOpenAI
import os
import streamlit.components.v1 as components
import re

# ✅ 환경 변수 설정
os.environ["openai_api_base"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

# ✅ 파일 경로 설정
csv_path = "./question20.csv"

# ✅ 데이터 로드
st.sidebar.header("📂 데이터 로딩 중...")
problems = load_html_explanation_data(csv_path)

# ✅ 사용자 정의 슬라이더 추가 (temperature 값 조절)
temperature = st.sidebar.slider("🌡️ GPT 창의성 조절 (Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# ✅ GPT 모델 설정
llm = AzureChatOpenAI(
    deployment_name="cats-aieng-prod-gpt4o-2024-05-13",
    openai_api_version="2024-05-01-preview",
    temperature=temperature
)

# ✅ `![이미지](URL)` 형식을 `<img>` 태그로 변환하는 함수
def convert_markdown_images_to_html(text):
    return re.sub(r"!\[(.*?)\]\((.*?)\)", r'<img src="\2" alt="\1" style="max-width: 100%; height: auto; display: block; margin: 10px auto;">', text)

# ✅ 문제에서 이미지 추출 함수 (문제에서만 이미지를 표시하고, 해설에서는 제외)
def extract_images_and_text(text):
    image_pattern = r"!\[.*?\]\((.*?)\)"
    images = re.findall(image_pattern, text)
    text_without_images = re.sub(image_pattern, "", text).strip()
    return images, text_without_images

# ✅ MathJax를 포함한 HTML 템플릿 (정렬 문제 해결을 위해 스타일 추가)
html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX 변환</title>
    <script>
        window.onload = function() {{
            if (window.MathJax) {{
                MathJax.typeset();
            }}
        }};
    </script>
    <script async src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            text-align: justify;
            margin: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: auto;
        }}
        h2 {{
            color: #1E88E5;
            border-bottom: 2px solid #1E88E5;
            padding-bottom: 5px;
        }}
        .content {{
            font-size: 18px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 10px;
        }}
        .image-container {{
            text-align: center;
            margin: 10px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="content">
            {converted_text}
        </div>
    </div>
</body>
</html>
"""

st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 모든 문제와 친절한 해설을 한 페이지에서 확인하세요!")

# ✅ 문제 & GPT 해설 출력
for index, problem in enumerate(problems):
    st.markdown(f"### 📝 문제 {index + 1} (ID: {problem['question_id']})")  # ✅ 문항아이디 포함

    # ✅ 문제에서 이미지 분리
    images, problem_text = extract_images_and_text(problem["question"])
    
    # ✅ 문제 출력 (이미지는 문제란에서만 표시)
    st.markdown("#### 🏫 문제")
    st.markdown(problem_text, unsafe_allow_html=True)
    
    for img in images:
        st.image(img, use_column_width=True)  # ✅ 문제에서만 이미지 출력

    # ✅ GPT 해설 생성
    with st.spinner(f"🔍 GPT가 문제 {index+1} 해설을 생성 중..."):
        detailed_explanation = generate_detailed_explanation(llm, problem_text, problem["explanation"])

    # ✅ 해설에서 이미지를 제거하고 정리
    detailed_explanation = convert_markdown_images_to_html(detailed_explanation)

    # ✅ MathJax와 이미지가 포함된 HTML 변환 적용
    rendered_html_explanation = html_template.format(converted_text=detailed_explanation)

    st.markdown("#### ✨ 해설")
    components.html(rendered_html_explanation, height=500)  # ✅ 해설을 깔끔한 정렬로 출력
