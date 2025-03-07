import os
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ✅ OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["deployment_name"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
os.environ["openai_api_base"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["openai_api_version"] = "2024-05-13"
os.environ["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY")

# ✅ (1) LaTeX 수식을 MathJax-friendly HTML로 변환
def convert_latex_to_mathjax(text):
    if not isinstance(text, str):
        return text  # 빈 데이터가 들어올 경우 그대로 반환
    
    latex_regex = re.compile(r'\\\((.*?)\\\)')  # \( ... \) 형태 감지
    
    def replace_latex(match):
        latex_code = match.group(1)
        return f'<span class="mathjax">\\({latex_code}\\)</span>'
    
    return latex_regex.sub(replace_latex, text)

# ✅ (2) HTML 형식의 문제 및 해설 데이터 로드 (LaTeX 변환 적용)
def load_html_explanation_data(file_path):
    df = pd.read_csv(file_path)  # CSV에서 HTML 형식 데이터 불러오기
    explanations = []
    for _, row in df.iterrows():
        explanations.append({
            "question_id": row["문항아이디"],  # ✅ 문항아이디 추가
            "question": convert_latex_to_mathjax(row["문제"]),
            "explanation": convert_latex_to_mathjax(row["해설"])
        })
    return explanations

# ✅ (3) GPT를 이용해 해설을 더 친절한 말투로 변환
def generate_detailed_explanation(llm, question, explanation):
    prompt_template = PromptTemplate(
        template=(
            "다음 문제의 해설을 초등학생도 이해할 수 있도록 친절하게 바꿔주세요:\n\n"
            "🔹 문제: {question}\n"
            "🔹 기존 해설: {explanation}\n\n"
            "💡 새로운 해설 (어떤 형식이든 사용자가 보기 편하게 모두 변환해서 출력해주세요.):"
        ),
        input_variables=["question", "explanation"]
    )

    response = llm.predict(prompt_template.format(question=question, explanation=explanation))
    return convert_latex_to_mathjax(response)  # 변환된 해설을 다시 LaTeX-friendly HTML로 변경
