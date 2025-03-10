import os
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# ✅ LaTeX 수식을 MathJax-friendly HTML로 변환
def convert_latex_to_mathjax(text):
    """
    LaTeX 수식을 HTML에서 MathJax로 렌더링 가능하도록 변환하는 함수
    """
    if not isinstance(text, str):
        return text  # 빈 데이터 방지

    # ✅ LaTeX 수식 감지하여 MathJax 변환
    latex_regex = re.compile(r'\\\((.*?)\\\)')

    def replace_latex(match):
        latex_code = match.group(1)
        return f'<span class="mathjax">\\({latex_code}\\)</span>'

    return latex_regex.sub(replace_latex, text)


# ✅ LLM을 이용해 친절한 해설을 생성하는 함수
def generate_detailed_explanation(llm, question, explanation, answer, user_prompt):
    """
    사용자가 입력한 문제, 해설, 정답을 바탕으로 AI가 친절한 해설을 생성하는 함수
    """
    prompt_template = PromptTemplate(
        template=(
            "{user_prompt}\n\n"
            "🔹 문제: {question}\n"
            "🔹 기존 해설: {explanation}\n"
            "🔹 정답: {answer}\n\n"
            "💡 AI가 변환한 해설:"
        ),
        input_variables=["user_prompt", "question", "explanation", "answer"]
    )

    response = llm.predict(prompt_template.format(
        user_prompt=user_prompt,
        question=question,
        explanation=explanation,
        answer=answer
    ))

    return response  # 변환된 해설을 다시 MathJax-friendly HTML로 변경
