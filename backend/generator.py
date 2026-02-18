"""
generator.py - Groq only. User adds GROQ_API_KEY in .env.
---------------------------------
✅ Only Groq API. Get key: https://console.groq.com/keys
"""

import os
import time
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

load_dotenv()

# Configuration
USE_API = os.getenv("USE_GROQ_API", "true").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Local fallback (unused — Groq only) — ONLY light models. Qwen 1.5B/1.7B = crash on low RAM, so we NEVER use them.
_LOCAL_ENV = os.getenv("LOCAL_MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")
if "Qwen" in _LOCAL_ENV or "1.5B" in _LOCAL_ENV or "1.7B" in _LOCAL_ENV:
    MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
    _OVERRIDE_QWEN = True  # so we can print once at startup
else:
    MODEL_NAME = _LOCAL_ENV
    _OVERRIDE_QWEN = False
# Minimum free RAM (MB) to allow loading local model. Below this we skip local and show API message.
MIN_RAM_MB_FOR_LOCAL = int(os.getenv("MIN_RAM_MB_FOR_LOCAL", "1800"))

def _available_ram_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return 2048.0  # assume enough if we can't detect

class Generator:
    def __init__(self, prompt_style="STRICT", api_key: str = None):
        self.prompt_style = prompt_style
        self.groq_client = None
        self.model = None
        self.tokenizer = None
        key = (api_key or "").strip() or GROQ_API_KEY
        self.use_api = USE_API and bool(key)

        if self.use_api:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=key)
                print("✅ Groq API ready.")
            except ImportError:
                print("⚠️ pip install groq")
                self.use_api = False
            except Exception as e:
                print(f"⚠️ Groq init failed: {e}")
                self.use_api = False
        if not self.use_api:
            if not key:
                print("⚠️ No Groq API key. Add in .env as GROQ_API_KEY or enter in app.")
            self.model = None
            self.tokenizer = None

    def build_prompt(self, query: str, docs: List[Union[str, Dict[str, Any]]], history: str = "") -> str:
        # 🎯 PROFESSIONAL PROMPT for Qwen2.5
        context_parts = []
        for i, doc in enumerate(docs[:3], 1):  # Top 3 docs for quality
            if isinstance(doc, dict):
                text = doc.get("text", "")
                # Good context size (500 chars per doc for quality)
                context_parts.append(f"[Document {i}]\n{text[:500]}")
            else:
                context_parts.append(f"[Document {i}]\n{str(doc)[:500]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer questions using the provided document context. Be conversational and helpful. Understand informal language and short forms. Only say "not in document" if the topic is completely unrelated.<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}

Answer helpfully from the context above.<|im_end|>
<|im_start|>assistant
"""
        
        return prompt

    def _get_top_score(self, docs: List[Union[str, Dict[str, Any]]], provided_score: float = None) -> float:
        if provided_score is not None:
            return provided_score
        
        if docs and len(docs) > 0:
            first_doc = docs[0]
            if isinstance(first_doc, dict):
                score = first_doc.get("score", first_doc.get("rerank_score", None))
                if score is not None:
                    return float(score)
        
        return 1.0

    def stream_generate(
        self,
        query: str,
        docs: List[Union[str, Dict[str, Any]]],
        history: str = "",
        top_score: float = None
    ):
        # Check if docs exist and have content
        if not docs:
            yield "No relevant documents found. Please upload a document first."
            return
        
        # Check if docs have actual text content
        has_content = False
        for doc in docs:
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            if text and text.strip():
                has_content = True
                break
        
        if not has_content:
            yield "The retrieved documents are empty. Please check your uploaded files."
            return

        prompt = self.build_prompt(query, docs, history)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,  # 🔥 Reduced from 2048 to 1024
            padding=True
        )

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # ⚡ SPEED-OPTIMIZED PARAMETERS
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=80,   # 🚀 Reduced to 80 for FAST response
            temperature=0.6,     # 🚀 Lower for faster, focused answers
            do_sample=False,     # 🚀 Greedy decoding = FASTER
            top_p=0.85,          
            repetition_penalty=1.1,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id
        )

        start = time.time()

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        token_count = 0
        
        for token in streamer:
            cleaned_token = token.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            if cleaned_token and cleaned_token.strip():
                token_count += 1
                yield cleaned_token

        thread.join()

        latency = time.time() - start
        actual_score = self._get_top_score(docs, top_score)
        print(f"⚡ TinyLlama: {latency:.2f}s | {token_count} tokens | Score: {actual_score:.2f}")
    
    def generate(
        self,
        query: str,
        docs: List[Union[str, Dict[str, Any]]],
        history: str = "",
        top_score: float = None
    ) -> str:
        # Check if docs exist and have content
        if not docs:
            return "No relevant documents found. Please upload a document first."
        
        # Check if docs have actual text content
        has_content = False
        for doc in docs:
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            if text and text.strip():
                has_content = True
                break
        
        if not has_content:
            return "The retrieved documents are empty. Please check your uploaded files."
        
        if not GROQ_API_KEY:
            return (
                "**Add your Groq API key to get answers.**\n\n"
                "1. Click **Get Groq API Key** in the sidebar (or go to https://console.groq.com/keys)\n"
                "2. Create your key (free)\n"
                "3. In `.env` add: `GROQ_API_KEY=your_key_here`\n"
                "4. Restart the app."
            )
        
        if self.groq_client is None and GROQ_API_KEY:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception:
                pass
        if self.groq_client is not None:
            return self._generate_with_groq(query, docs, history)
        
        return (
            "**Groq API could not be used.** Check your `GROQ_API_KEY` in `.env` and restart. "
            "Get key: https://console.groq.com/keys"
        )

    def _build_api_messages(self, query: str, docs: List[Union[str, Dict[str, Any]]], history: str = ""):
        """Build (messages, is_summary_question) for Groq/OpenRouter. Shared by both APIs."""
        query_lower = query.lower()
        followup_markers = [
            "anything more", "anything else", "tell me more",
            "more about", "more on", "aur batao", "aur btao", "aur?", "more?"
        ]
        is_followup_more = any(m in query_lower for m in followup_markers)
        summary_markers = [
            "summarise", "summarize", "summary", "overview",
            "tldr", "tl;dr", "gist", "smrz", "summ",
            "entire document", "whole document", "complete document",
            "entire pdf", "whole pdf", "full document",
        ]
        is_summary_question = (
            not is_followup_more and any(word in query_lower for word in summary_markers)
        )
        MAX_CONTEXT_CHARS = 25000 if is_summary_question else 8000
        MAX_CHARS_PER_CHUNK = 800 if is_summary_question else 600
        num_docs = len(docs) if is_summary_question else min(5, len(docs))
        context_parts = []
        total_chars = 0
        seen_texts = set()
        for i, doc in enumerate(docs[:num_docs], 1):
            text = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            text = text.strip()
            if not text:
                continue
            text_key = text[:200].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            chunk_text = text[:MAX_CHARS_PER_CHUNK]
            if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
                break
            context_parts.append(f"[Section {i}]\n{chunk_text}")
            total_chars += len(chunk_text)
        context = "\n\n".join(context_parts)
        print(f"📝 Context: {len(context_parts)} unique sections, {total_chars} chars")
        if is_summary_question:
            system_prompt = """You are a smart, friendly AI assistant — like ChatGPT. You analyze documents naturally.

FOR SUMMARIES (including questions like "iss doc mein kya hai?", "what is this document about?", "summarise this doc"):
1. Read ALL sections carefully — don't miss any topic
2. Identify EVERY distinct topic/subject in the document (this might be:
   - a research paper, a resume/CV, a bill/invoice, a legal contract, a scientific report, a textbook chapter, meeting notes, etc.)
3. Group related information together
4. Give equal coverage to ALL topics found

RESPONSE FORMAT:
Start with a friendly one-liner about what the document covers overall.
Then list EACH topic as a separate section with a bold header and 2-3 bullet points.

Example tone:
"This document covers several interesting topics! Here's what I found:"

**Topic 1 Name**
- Key point
- Key point

**Topic 2 Name**  
- Key point
- Key point

If there are important NUMERICAL values (e.g. EQE %, salary, prices, measurements), include them in the relevant section with correct labels.

CRITICAL RULES:
- Cover EVERY topic found, not just the first one
- If there are 5 different subjects, mention all 5
- Don't say "the document repeats information" — instead extract unique content
- Keep it readable and conversational, not robotic
- Use simple language anyone can understand"""
        else:
            system_prompt = """You are a smart, friendly AI assistant — like ChatGPT. You help users understand their uploaded documents.

PERSONALITY:
- Natural, conversational, warm — never robotic or overly formal
- Understand informal language, typos, short forms, Hinglish ("kya h ye", "btao", "smrz kro").
- Map short forms to full terms when the document uses the full form: "elastic" → Elasticsearch, "oled" → OLEDs, "ai" → artificial intelligence, "jd" → job description. Answer about the full concept when the context contains it.
- Keep responses concise and to the point

FOR SHORT/GREETING MESSAGES (like "hi", "hy", "hello", "hey"):
- Respond briefly and warmly: "Hey! I've read your document(s). What would you like to know?" (or "What would you like to know about it?" if a single doc). Don't say anything like "no relevant documents" or "upload documents" — the user already has docs loaded.
- Don't over-explain or list all topics unless asked.
- Keep it to 1-2 sentences max.

FOR QUESTIONS:
- Documents can be ANY type (resume, JD, research paper, invoice, contract, report, handbook, etc.). Adapt your answer to what the context contains.
- First, infer what KIND of document(s) you have from the context. If the conversation history shows the user previously asked about a specific doc or topic (e.g. "the resume", "that PDF", "elastic"), answer about THAT same doc/topic using the context that matches it. If the user is asking a new question with no prior topic, prefer the most recent or most relevant part of the context.
- Always answer using ONLY the provided context sections. Do not rely on outside knowledge.
- Give a direct, clear answer first, grounded in the document.
- If the document clearly defines one concept matching the question (e.g. "Elasticsearch", "OLEDs", a specific ROLE / JOB DESCRIPTION, or a specific section of an invoice/contract), explain THAT directly — don't invent alternative meanings.
- For questions about roles / jobs (phrases like "this role", "job description", "JD", "what is this role about", "responsibilities", "key responsibilities"):
    - First, locate the part of the context that looks like a job description (title, responsibilities, requirements, location, team, etc.).
    - Summarize the role in 2–4 bullet points, focusing on what the role does, key responsibilities, and any requirements mentioned.
    - Do NOT answer with generic company culture or benefits if a concrete role/job section exists in the context.
- For follow-up questions like "anything more?", "tell me more", "aur batao", "anything more about X", ALWAYS use the conversation history to detect what the user was just asking about (e.g. Elasticsearch, OLEDs, EQE value, salary, THIS ROLE, invoice amount, policy clause, etc.) and continue talking ONLY about that SAME topic. Do NOT suddenly summarize other, unrelated topics from the document, even if they appear in the context.
- If the user asks "anything more about [topic]" (e.g. "anything more about elastic?"), answer ONLY about that topic. Treat "elastic" as Elasticsearch, "oled" as OLEDs, etc. If you already answered about that topic in the previous message, give additional details from the context about that same topic. NEVER say "[topic] is not in the document" if the conversation history shows you already answered about it or if the context contains related wording (e.g. "Elasticsearch" when user said "elastic").
- Only bring in additional interpretations if the document actually talks about them and they are part of the same topic thread.
- Add bullet points only if the answer has multiple parts.
- Use simple language.
- Don't repeat the question back.
- Don't add unnecessary disclaimers like "you might be referring to something else".

NUMBER / VALUE SAFETY (VERY IMPORTANT):
- Many questions will ask for specific numerical values (e.g. EQE of red OLED, efficiency %, voltages, prices, counts, etc.).
- When answering value questions, ONLY use numbers that are explicitly present in the provided context.
- Never guess or average values from memory — if the exact value for the requested variant (e.g. "red OLED") is not clearly visible in the context, say that it is unclear from the document.
- If the document lists MULTIPLE values (e.g. EQE for blue, green, red), always label them clearly (e.g. "Red OLED EQE: 17.3%, Green: 21.5%") and make sure you don't mix them up.
- If the question asks for a specific variant (e.g. "red OLED") and you only see values for other variants, explain that those other values exist but the requested one is not shown.

WHEN TO SAY "NOT IN DOCUMENT":
- Before saying this, carefully scan the context for any sentences that mention the main keywords from the question.
- ONLY say "not in document" if there is truly no overlap at all (e.g., asking about cricket in a medical report).
- If there is even partial information related to the question (for example, any description of the role, even if brief), answer using that partial information instead of claiming it is not described."""
        if is_summary_question:
            user_msg = f"""Here are all the sections from the uploaded document:

{context}

User's request: {query}

Summarize this document covering EVERY distinct topic you find. Don't skip any subject."""
        else:
            history_block = f"\n\nRecent conversation:\n{history}\n\n" if (history and history.strip()) else ""
            user_msg = f"""Document context:

{context}
{history_block}User says: {query}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        return (messages, is_summary_question)

    def _generate_with_groq(self, query: str, docs: List[Union[str, Dict[str, Any]]], history: str = "") -> str:
        """Ultra-fast generation using Groq API (1-2s response time!)"""
        try:
            messages, is_summary_question = self._build_api_messages(query, docs, history)
            start = time.time()
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4,
                max_tokens=1200 if is_summary_question else 600,
                top_p=0.95,
                presence_penalty=0.1,
            )
            latency = time.time() - start
            answer = completion.choices[0].message.content.strip()
            print(f"⚡ Groq API: {latency:.2f}s | {len(answer)} chars | High quality!")
            return answer
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            err = str(e).lower()
            if "429" in err or "rate" in err:
                return "**Groq rate limit (429).** Wait a while or use a new API key. Get key: https://console.groq.com/keys"
            return f"**Groq API error.** {e} Check key: https://console.groq.com/keys"

    def _generate_local(self, query: str, docs: List[Union[str, Dict[str, Any]]], history: str = "", top_score: float = None) -> str:
        """Not used when Groq-only; kept for reference."""
        if self.tokenizer is None or self.model is None:
            return (
                "**Add your Groq API key to get answers.**\n\n"
                "1. Click **Get Groq API Key** in the sidebar or go to https://console.groq.com/keys\n"
                "2. Create your key (free)\n"
                "3. In `.env` add: `GROQ_API_KEY=your_key_here`\n"
                "4. Restart the app."
            )
        
        prompt = self.build_prompt(query, docs, history)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        start = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # More tokens for comprehensive answers
                do_sample=True,      # Sampling for better quality
                temperature=0.6,     # Balanced for quality
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        latency = time.time() - start
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|im_start|>assistant" in answer:
            answer = answer.split("<|im_start|>assistant")[-1]
        
        answer = answer.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        
        token_count = len(outputs[0])
        actual_score = self._get_top_score(docs, top_score)
        print(f"⚡ Local mode: {latency:.2f}s | {token_count} tokens | Score: {actual_score:.2f}")
        
        return answer


if __name__ == "__main__":
    print("Testing SPEED-OPTIMIZED generator...")
    
    gen = Generator()
    
    test_docs = [
        {
            "text": "OLEDs are organic light-emitting diodes used in displays. They emit light when electricity passes through organic compounds.",
            "score": 8.5,
            "source": "oled.pdf"
        }
    ]
    
    print("\n🔸 Speed test: What are OLEDs?")
    start = time.time()
    answer = ""
    for token in gen.stream_generate("What are OLEDs?", test_docs):
        answer += token
        print(token, end="", flush=True)
    total_time = time.time() - start
    print(f"\n\n⏱️ Total time: {total_time:.2f}s")
    print(f"📝 Answer length: {len(answer)} chars")
    
    if total_time < 3.0:
        print("✅ SPEED TEST PASSED! Under 3 seconds!")
    else:
        print("⚠️ Still slow. Try TinyLlama for even faster responses.")