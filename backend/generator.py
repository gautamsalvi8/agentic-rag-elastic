from transformers import pipeline
import time


class Generator:
    def __init__(self, prompt_style="STRICT"):

        self.prompt_style = prompt_style

        # ✅ TinyLlama chat model
        self.llm = pipeline(
            task="text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=150,
            do_sample=False,              # deterministic answers
            return_full_text=False        # ⭐ prevents prompt echo
        )

    # --------------------------------------------------
    # Prompt Builder
    # --------------------------------------------------
    def build_prompt(self, query, docs):

        context = "\n\n".join(docs)

        # ⭐ Chat style works MUCH better for TinyLlama
        if self.prompt_style == "STRICT":
            prompt = f"""<|system|>
You are a helpful AI assistant.
Answer ONLY using the provided context.
If answer is not present, say "Not found".
<|user|>
Context:
{context}

Question: {query}
<|assistant|>
"""
        else:
            prompt = f"""<|user|>
Context:
{context}

Question: {query}
<|assistant|>
"""

        return prompt

    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------
    def generate(self, query, docs):

        prompt = self.build_prompt(query, docs)

        start = time.time()

        output = self.llm(prompt)

        latency = time.time() - start

        answer = output[0]["generated_text"].strip()

        # simple token estimate
        approx_tokens = len(prompt.split()) + len(answer.split())

        print(f"\n📊 Tokens used (approx): {approx_tokens}")
        print(f"⚡ LLM latency: {round(latency, 3)}s")

        return answer
