import os
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

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
def generate_question(llm, question):
    prompt_template = PromptTemplate(
        template=("사용자가 읽을 때 가독성이 좋도록 문장별로 줄바꿈이나 띄어쓰기 등을 잘 지켜주세요.\n\n"
            "이미지 url들은 모두 제외하고 출력해주세요.\n\n"
            "문제에 대한 풀이는 절대 출력하지 말아주세요.\n\n"
            "🔹 문제: {question}"
        ),
        input_variables=["question"]
    )

    response0 = llm.predict(prompt_template.format(question=question))
    return convert_latex_to_mathjax(response0)


# ✅ (3) GPT를 이용해 해설을 더 친절한 말투로 변환
def generate_detailed_explanation(llm, explanation):
    prompt_template = PromptTemplate(
        template=("다음 문제의 해설을 초등학생도 이해할 수 있도록 친절하게 바꿔주세요:\n\n"
            "해설을 바꿔주겠다는 대답은 따로 안해줘도 될 것 같아요.\n\n"
            "사용자가 읽을 때 가독성이 좋도록 문장별로 줄바꿈이나 띄어쓰기 등을 잘 지켜주세요.\n\n"
            "🔹 기존 해설: {explanation}\n\n"
            "💡 새로운 해설 (어떤 형식이든 사용자가 보기 편한 양식으로 변환해주세요.):"
        ),
        input_variables=["explanation"]
    )

    response = llm.predict(prompt_template.format(explanation=explanation))
    
    # ✅ GPT가 생성한 해설도 다시 LaTeX 변환 적용
    return convert_latex_to_mathjax(response)
