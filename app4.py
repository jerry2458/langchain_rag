import streamlit as st
from rag_functions4 import load_html_explanation_data, generate_detailed_explanation
from langchain.chat_models import AzureChatOpenAI
import os
import streamlit.components.v1 as components
import re

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

def extract_image_from_text(text):
    """
    HTML 태그 <img src="..."> 에서 이미지 URL만 추출하여 반환
    """
    image_pattern = r'<img[^>]+src="([^">]+)"'  # ✅ HTML <img src="URL">에서 URL만 추출
    images = re.findall(image_pattern, text)  # ✅ 모든 이미지 URL 리스트 추출
    
    # ✅ 이미지 태그 제거한 텍스트 반환
    text_without_images = re.sub(image_pattern, "", text).strip()
    
    return images, text_without_images


# ✅ MathJax 스크립트 추가 (LaTeX 수식 렌더링)
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
            max-width: 2000px;
            margin: auto;
        }}
        h2 {{
            color: #1E88E5;
            border-bottom: 2px solid #1E88E5;
            padding-bottom: 5px;
        }}
        .content {{
            font-size: 12px;
            padding: 12px;
            background: #f9f9f9;
            border-radius: 10px;
            white-space: pre-line;  /* ✅ 줄바꿈 유지 */
            word-wrap: break-word;  /* ✅ 긴 단어 줄바꿈 */
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


# st.components.v1.html(mathjax_script, height=0)

st.title("📘 AI 수학 문제 해설 도우미")
st.write("📢 모든 문제와 친절한 해설을 한 페이지에서 확인하세요!")

# ✅ 문제 & GPT 해설 출력
for index, problem in enumerate(problems):
    st.markdown(f"### 📝 문제 {index + 1} (ID: {problem['question_id']})")  # ✅ 문항아이디 포함

    # ✅ 문제에서 이미지 추출하여 문항 ID 아래에 먼저 표시
    images, problem_text = extract_image_from_text(problem["question"])
    
    for img in images:
        st.image(img)  # ✅ 문제에서 추출된 이미지 바로 출력
    
    with st.spinner(f"🔍 GPT가 문제 {index+1} 해설을 생성 중..."):
        detailed_explanation = generate_detailed_explanation(llm, problem["question"], problem["explanation"])
    
    
    rendered_html_explanation = html_template.format(converted_text=detailed_explanation)
    ㅊ = max(200, len(rendered_html_explanation) // 3)
    st.markdown("#### ✨ 문제와 해설")
    components.html(rendered_html_explanation, height=estimated_height)
    # st.markdown(rendered_html_explanation, unsafe_allow_html=True)  # ✅ GPT 변환 해설 출력
