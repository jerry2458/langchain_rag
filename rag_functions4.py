import os
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ✅ OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ (1) LaTeX 수식을 MathJax-friendly HTML로 변환
def convert_latex_to_mathjax(text):
    if not isinstance(text, str):
        return text  # 빈 데이터는 그대로 반환

    # ✅ \(\) 또는 $$ $$ 블록을 MathJax-friendly HTML로 변환
    text = re.sub(r'\\\((.*?)\\\)', r'<span class="mathjax">\\(\1\\)</span>', text)  # 인라인 수식
    text = re.sub(r'\\\[(.*?)\\\]', r'<div class="mathjax">\\[\1\\]</div>', text)  # 블록 수식

    # ✅ \displaystyle, \text, \boxed{} 처리
    text = text.replace("\\displaystyle", "").replace("\\text", "").replace("\\mathstrut", "")
    text = re.sub(r'\\boxed\{(.*?)\}', r'<span class="mathjax">\\(\1\\)</span>', text)

    return text

# ✅ (2) HTML 형식의 문제 및 해설 데이터 로드 (LaTeX 변환 적용)
def load_html_explanation_data(file_path):
    df = pd.read_csv(file_path)  # CSV에서 HTML 형식 데이터 불러오기
    explanations = []
    for _, row in df.iterrows():
        explanations.append({
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
            "💡 새로운 해설 (HTML 형식 유지, LaTeX 수식은 MathJax로 변환해서 출력):"
        ),
        input_variables=["question", "explanation"]
    )

    response = llm.predict(prompt_template.format(question=question, explanation=explanation))
    return convert_latex_to_mathjax(response)  # 변환된 해설을 다시 LaTeX-friendly HTML로 변경
