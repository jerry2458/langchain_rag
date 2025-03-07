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

# ✅ (3) GPT를 이용해 문제를 정리하는 함수
def refine_question(llm, question):
    prompt_template = PromptTemplate(
        template=(
           "다음 수학 문제를 웹에서 보기 편한 형식으로 변환해 주세요."
        "수식을 깔끔하게 정리하고, 필요할 경우 단위를 명확히 표시하세요."
        "html 태그가 코드처럼 보이지 않고 실제로 렌더링 되도록 해주세요."
        "불필요한 LaTeX 명령어는 제거하고, MathJax를 사용하여 웹에서 잘 보이도록 변환해 주세요.\n\n"
        "{question}\n\n"
        "💡 변환된 문제 (어떤 형식이든 사용자가 보기 편한 양식으로 변환해주세요.):"
        ),
        input_variables=["question"]
    )

    response = llm.predict(prompt_template.format(question=question))
    return convert_latex_to_mathjax(response)  # 변환된 문제를 다시 MathJax-friendly HTML로 변경

# ✅ (4) GPT를 이용해 해설을 친절하고 상세하게 변환하는 함수
def refine_explanation(llm, explanation):
    prompt_template = PromptTemplate(
        template=(
             "다음 해설을 웹에서 보기 쉽게 변환하고, 초등학생도 이해할 수 있도록 친절하고 상세하게 설명해주세요. "
        "html 태그가 코드처럼 보이지 않고 실제로 렌더링 되도록 해주세요."
        "불필요한 LaTeX 명령어는 제거하고, MathJax를 사용하여 웹에서 잘 보이도록 변환해 주세요.\n\n"
        "{explanation}\n\n"
        "💡 변환된 해설 (어떤 형식이든 사용자가 보기 편한 양식으로 변환해주세요.):"
        ),
        input_variables=["explanation"]
    )

    response = llm.predict(prompt_template.format(explanation=explanation))
    return convert_latex_to_mathjax(response)  # 변환된 해설을 다시 MathJax-friendly HTML로 변경
