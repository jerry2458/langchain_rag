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
            f"🔹 문제: {question}\n"
            f"🔹 기존 해설: {explanation}\n"
            f"🔹 정답: {answer}\n"
            f"{user_prompt}"
            "출력 데이터 예시 (참고용)\n"
            "💡 AI가 변환한 해설 : <p class="para0" style="text-align:left;color:black;">이 문제는 정육면체에 대한 것입니다. 정육면체는 우리가 흔히 알고 있는 주사위 모양의 입체 도형입니다. 이 도형에는 꼭짓점, 면, 그리고 모서리가 있습니다. 여기서 우리는 보이지 않는 꼭짓점의 수와 보이는 면의 수를 알아내야 합니다.</p><p class="para0" style="text-align:left;color:black;">먼저, 정육면체의 꼭짓점에 대해 알아볼까요? 정육면체는 총 <span data-mathinfo="86,1000,525,975;;8" class="itexmath" contenteditable="false" data-latex="\displaystyle  8 ">\(\displaystyle 8 \)</span>개의 꼭짓점을 가지고 있습니다. 하지만 문제에서는 보이지 않는 꼭짓점의 수를 묻고 있습니다. 보이지 않는 꼭짓점은 정육면체의 뒷면에 있는 꼭짓점으로, 우리가 정육면체를 바라볼 때는 볼 수 없습니다. 이 경우 보이지 않는 꼭짓점은 <span data-mathinfo="86,1000,525,975;;1" class="itexmath" contenteditable="false" data-latex="\displaystyle  1 ">\(\displaystyle 1 \)</span>개입니다.</p><p class="para0" style="text-align:left;color:black;">다음으로 보이는 면의 수를 살펴보겠습니다. 정육면체는 총 <span data-mathinfo="86,1000,525,975;;6" class="itexmath" contenteditable="false" data-latex="\displaystyle  6 ">\(\displaystyle 6 \)</span>개의 면을 가지고 있습니다. 하지만 우리가 보고 있는 면은 정육면체의 앞면, 왼쪽 면, 그리고 위쪽 면으로, 이 경우 보이는 면은 <span data-mathinfo="86,1000,525,975;;3" class="itexmath" contenteditable="false" data-latex="\displaystyle  3 ">\(\displaystyle 3 \)</span>개입니다.</p><p class="para0" style="text-align:left;color:black;">이제 보이지 않는 꼭짓점의 수와 보이는 면의 수를 더해보겠습니다. 보이지 않는 꼭짓점이 <span data-mathinfo="86,1000,525,975;;1" class="itexmath" contenteditable="false" data-latex="\displaystyle  1 ">\(\displaystyle 1 \)</span>개이고, 보이는 면이 <span data-mathinfo="86,1000,525,975;;3" class="itexmath" contenteditable="false" data-latex="\displaystyle  3 ">\(\displaystyle 3 \)</span>개이므로, 이 둘을 더하면 <span data-mathinfo="86,1000,3925,975;;1+3 =4" class="itexmath" contenteditable="false" data-latex="\displaystyle  1+3 =4 ">\(\displaystyle 1+3 =4 \)</span>개가 됩니다.</p><p class="para0" style="text-align:left;color:black;">따라서, 보이지 않는 꼭짓점의 수와 보이는 면의 수의 합은 <span data-mathinfo="86,1000,525,975;;4" class="itexmath" contenteditable="false" data-latex="\displaystyle  4 ">\(\displaystyle 4 \)</span>입니다.</p>"
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

    return response  
