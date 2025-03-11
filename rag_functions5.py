import re
from langchain.prompts import PromptTemplate

# ✅ LLM을 이용해 친절한 해설을 생성하는 함수
def generate_detailed_explanation(llm, question, explanation, answer, user_prompt):
    """
    사용자가 입력한 문제, 해설, 정답을 바탕으로 AI가 친절한 해설을 생성하는 함수
    """
    prompt_template = PromptTemplate(
        template=(
            "다음은 HTML 태그가 포함된 수학 문제, 해설, 정답 데이터입니다:\n"
            "🔹 문제: {question}\n"
            "🔹 기존 해설: {explanation}\n"
            "🔹 정답: {answer}\n"
            "{user_prompt}\n"
            "💡 AI가 변환한 해설 :"
        ),
        input_variables=["user_prompt", "question", "explanation", "answer"]
    )

    response = llm.predict(prompt_template.format(
        user_prompt=user_prompt,
        question=question,
        explanation=explanation,
        answer=answer
        )
    )

    return str(response)  
