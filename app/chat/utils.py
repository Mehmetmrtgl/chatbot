from scripts.db_utils import get_answer_from_db, get_chat_history, get_all_questions
from scripts.rag_utils import get_context_from_faiss
from scripts.faiss_utils import get_similar_questions, get_similar_questions_hybrid
from scripts.model_inference import generate_answer
from scripts.finetune_lookup import is_in_finetune_data
from app.db.models import ChatLog, db, PendingQuestion
from sentence_transformers import SentenceTransformer
from datetime import datetime
import unicodedata
import re

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SOHBET_BITIRICI_KELIMELER = [
    "tesekkur", "tesekkurler", "sag ol", "sagol", "tamam", "anladim", "cok iyi",
    "eyvallah", "gorusuruz", "bye", "hosca kal", "okey", "oldu", "simdilik bu kadar"
]

BAD_PATTERNS = [
    r"maalesef.*bilgi.*ulaşamıyorum",
    r"sorunuzla ilgili.*bilgi.*yok",
    r"verilen bilgiler.*cevaplayamıyorum",
    r"ilgili.*içerik.*bulunamadı",
    r"konuyla ilgili.*bilgiye ulaşamadım",
    r"sorunuzla.*yardımcı olamıyorum",
]

def normalize_input(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_exit_phrase(text: str) -> bool:
    normalized = normalize_input(text)
    return any(phrase in normalized for phrase in SOHBET_BITIRICI_KELIMELER)

def log_chat_message(role, message, session_id):
    log = ChatLog(role=role, message=message, session_id=session_id)
    db.session.add(log)
    db.session.commit()

def context_overlap_score(context: str, answer: str) -> float:
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())
    overlap = context_words.intersection(answer_words)
    return len(overlap) / max(len(context_words), 1)

def is_hallucinated(answer: str) -> bool:
    for pattern in BAD_PATTERNS:
        if re.search(pattern, answer.lower()):
            return True
    return False

def evaluate_rag_answer(user_question, rag_context_short, final_answer, model_name="llama"):
    if len(final_answer.split()) < 5:
        return "low"
    if is_hallucinated(final_answer):
        return "low"
    if context_overlap_score(rag_context_short, final_answer) < 0.15:
        return "medium"

    evaluation_prompt = (
        f"Aşağıda bir soru, bağlam ve cevap var.\n\n"
        f"Soru: {user_question}\n"
        f"Bağlam: {rag_context_short}\n"
        f"Cevap: {final_answer}\n\n"
        "Bu cevabı değerlendir. Sadece 'Uygun', 'Orta' veya 'Yetersiz' şeklinde cevap ver:"
    )

    llm_evaluation = generate_answer(evaluation_prompt, model_name=model_name).lower()
    print("🧠 LLM değerlendirmesi:", llm_evaluation)

    if "yetersiz" in llm_evaluation:
        return "low"
    elif "orta" in llm_evaluation:
        return "medium"
    else:
        return "high"

def answer_user_question(user_question: str, session_id: str = "default_user", chat_history=None):
    normalized_input = normalize_input(user_question)

    if normalized_input in ["merhaba", "selam", "nasilsin", "iyi gunler"]:
        greeting_response = "Merhaba! Size nasıl yardımcı olabilirim? Kütüphane ile ilgili sorularınızı sorabilirsiniz."
        log_chat_message("assistant", greeting_response, session_id)
        return {"status": "greeting", "answer": greeting_response}

    if is_exit_phrase(user_question):
        farewell = "Rica ederim, görüşmek üzere!"
        log_chat_message("assistant", farewell, session_id)
        print("🟠 Sohbet bitirici ifade algılandı.")
        return {"status": "ended", "answer": farewell}

    log_chat_message("user", user_question, session_id)
    print(f"Normalized input: {normalized_input}")

    db_answer = get_answer_from_db(normalized_input)
    if db_answer:
        log_chat_message("assistant", db_answer["answer"], session_id)
        return {"status": "from_db", "answer": db_answer, "question_id": db_answer["id"]}

    if is_in_finetune_data(normalized_input):
        answer = generate_answer(user_question, original_question=user_question)
        log_chat_message("assistant", answer, session_id)
        return {"status": "from_finetune", "answer": answer}

    rag_context = get_context_from_faiss(user_question)
    rag_context_short = "\n\n".join(rag_context.split("\n\n")[:1]).strip()

    if not rag_context_short:
        all_q = get_all_questions()
        hybrid_suggestions = get_similar_questions_hybrid(normalized_input, all_q)
        top_text = hybrid_suggestions[0] if hybrid_suggestions else ""
        fallback_msg = "Bu soruya doğrudan bir cevap bulunamadı."
        if top_text:
            fallback_msg += f" Bunu mu demek istediniz: {top_text}"
        log_chat_message("assistant", fallback_msg, session_id)
        return {
            "status": "no_answer",
            "message": fallback_msg,
            "suggestions": [top_text] if top_text else []
        }

    if chat_history is None:
        chat_history = get_chat_history(session_id)
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in chat_history])

    prompt = (
        "Kütüphane ile ilgili kullanıcı sorularına yanıt ver. "
        "Yanıtta sadece verilen bağlam bilgisini ve önceki sohbeti kullan. "
        "Yanıt tamamen **Türkçe** olmali, İngilizce kelime veya cümle **kullanma**. "
        "Konu dışı sorular için 'Bu konuda bir fikrim yok.' yaz. "
        # "Yanıtı maksimum 3 kısa cümleyle sınırla.\n\n"
        f"Önceki Sohbet:\n{history_text}\n\n"
        f"Bağlam:\n{rag_context_short}\n\n"
        f"Soru: {user_question}\nCevap:"
    )

    final_answer = generate_answer(prompt, original_question=user_question)
    quality = evaluate_rag_answer(user_question, rag_context_short, final_answer)

    if quality == "low":
        similar_questions = get_similar_questions(normalized_input)
        suggestion = similar_questions[0] if similar_questions else ""
        msg = "Bu konuda emin değilim. Şunu mu demek istediniz?"
        log_chat_message("assistant", msg, session_id)
        return {
            "status": "low_quality_rag",
            "answer": msg,
            "suggestions": [suggestion] if suggestion else []
        }

    pending_entry = PendingQuestion(
        question=user_question,
        answer=final_answer,
        suggested_source="llm_rag"  
    )
    db.session.add(pending_entry)
    db.session.commit()

    log_chat_message("assistant", final_answer, session_id)


    return {
        "status": "from_rag",
        "answer": final_answer,
        "question_id": pending_entry.id, 
        "context": rag_context_short,
        "quality": quality
    }
