import os
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ✅ (1) LaTeX 수식을 MathJax-friendly HTML로 변환
# def convert_latex_to_mathjax(text):
#     if not isinstance(text, str):
#         return text  # 빈 데이터가 들어올 경우 그대로 반환
    
#     latex_regex = re.compile(r'\\\((.*?)\\\)')  # \( ... \) 형태 감지
    
#     def replace_latex(match):
#         latex_code = match.group(1)
#         return f'<span class="mathjax">\\({latex_code}\\)</span>'
    
#     return latex_regex.sub(replace_latex, text)

def convert_latex_to_mathjax(text):
    if not isinstance(text, str):
        return text  # 빈 데이터는 그대로 반환

    # ✅ \text{내용} 제거 (내용만 남기고 변환)
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)

    # ✅ \( ... \) 인라인 수식 변환
    text = re.sub(r'\\\((.*?)\\\)', r'<span class="mathjax">\\(\1\\)</span>', text)
    
    # ✅ \[ ... \] 블록 수식 변환
    text = re.sub(r'\\\[(.*?)\\\]', r'<div class="mathjax">\\[\1\\]</div>', text)

    # ✅ $$ ... $$ 블록 수식 변환
    text = re.sub(r'\$\$(.*?)\$\$', r'<div class="mathjax">\\[\1\\]</div>', text)

    # ✅ 수학 기호 변환 (\pi, \times, \approx 등)
    text = text.replace("\\pi", "π")
    text = text.replace("\\times", "×")
    text = text.replace("\\approx", "≈")

    # ✅ 불필요한 LaTeX 명령어 제거
    text = text.replace("\\displaystyle", "").replace("\\mathstrut", "")

    # ✅ \boxed{} 변환
    text = re.sub(r'\\boxed\{(.*?)\}', r'<span class="mathjax">\\(\1\\)</span>', text)

    return text



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
           "LaTeX 수식은 MathJax로 변환해서 출력해주세요."
           "어떤 형식이든 사용자가 보기 편하게 모두 변환해서 출력해주세요."
           "🔹 문제: {question}\n"
           "💡 변환된 문제 (어떤 형식이든 사용자가 보기 편하게 모두 변환해서 출력해주세요.):"
        ),
        input_variables=["question"]
    )

    response = llm.predict(prompt_template.format(question=question))
    return convert_latex_to_mathjax(response)  # 변환된 문제를 다시 MathJax-friendly HTML로 변경

# ✅ (4) GPT를 이용해 해설을 친절하고 상세하게 변환하는 함수
def refine_explanation(llm, explanation):
    prompt_template = PromptTemplate(
        template=(
             "다음 해설을 초등학생도 이해할 수 있도록 친절하고 상세하게 설명해주세요."
             "LaTeX 수식은 MathJax로 변환해서 출력해주세요."
             "어떤 형식이든 사용자가 보기 편하게 모두 변환해서 출력해주세요."
             "🔹 해설: {explanation}\n\n"
             "💡 변환된 해설 (어떤 형식이든 사용자가 보기 편한 양식으로 변환해주세요.):"
        ),
        input_variables=["explanation"]
    )

    response = llm.predict(prompt_template.format(explanation=explanation))
    return convert_latex_to_mathjax(response)  # 변환된 해설을 다시 MathJax-friendly HTML로 변경
