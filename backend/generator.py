from transformers import pipeline
import time


class Generator:
    def __init__(self, prompt_style="SMART"):
        self.prompt_style = prompt_style

        self.llm = pipeline(
            task="text2text-generation",
            model="google/flan-t5-small",
            temperature=0.3,
            max_new_tokens=512
        )

    # =============================
    # 🔥 BETTER PROMPT ENGINEERING
    # =============================
    def build_prompt(self, query, docs, history=""):
        context = "\n\n".join(docs[:3])

        prompt = f"""
You are a smart document assistant.

STYLE:
• concise but informative
• use bullets or short paragraphs
• beginner friendly if asked
• summarize instead of copying
• highlight key terms
• avoid fluff

RULES:
• ONLY use context
• if missing → say "Not found in documents"

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

        return prompt

    # =============================
    # GENERATION
    # =============================
    def generate(self, query, docs, history=""):
        prompt = self.build_prompt(query, docs, history)

        start = time.time()

        output = self.llm(
            prompt,
            num_beams=4,
            early_stopping=True,
            min_length=30,
            max_length=400
        )

        latency = time.time() - start

        answer = output[0]["generated_text"].strip()

        print(f"⚡ LLM latency: {latency:.2f}s")

        return answer


def generate_answer(query, docs, history_context=""):
    gen = Generator()
    return gen.generate(query, docs, history_context)
