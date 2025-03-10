from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

def generate_detailed_explanation(llm, question, explanation, answer, user_prompt):
    """
    사용자가 입력한 문제, 해설, 정답을 바탕으로 AI가 친절한 해설을 생성하는 함수.
    """

    # ✅ PromptTemplate을 문자열로 변환
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

    formatted_prompt = prompt_template.format(
        user_prompt=user_prompt,
        question=question,
        explanation=explanation,
        answer=answer
    )

    # ✅ LangChain의 generate()에 전달될 올바른 형식으로 변환
    response = llm.generate([[HumanMessage(content=formatted_prompt)]])

    return response.generations[0][0].text  # ✅ 응답 객체에서 텍스트만 추출
