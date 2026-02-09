import time
from transformers import pipeline

generator = pipeline(
    "text2text-generation",   # 🔥 FIXED
    model="google/flan-t5-base",
    device=-1
)

def ask_llm(question, context):

    prompt = f"""
Use the context to answer the question.
If context is insufficient, give a short general answer.

Context:
{context}

Question: {question}
Answer:
"""


    start = time.time()

    output = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False
    )

    latency = round(time.time() - start, 3)

    return output[0]["generated_text"], latency
