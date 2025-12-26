import json
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_PARAMS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "hu_chatbot2"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "")
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "../data/manual_questions.json")

def insert_questions():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Dosya bulunamadı: {JSON_PATH}")
        return

    print(f"📂 {JSON_PATH} okunuyor...")
    
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON okuma hatası: {e}")
        return

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        added_count = 0
        updated_count = 0

        for item in data:
            question = item.get("question")
            answer_raw = item.get("answer")

            if isinstance(answer_raw, list):
                answer = " ".join(answer_raw)
            elif isinstance(answer_raw, str):
                answer = answer_raw
            else:
                continue 

            if question and answer:
                question = question.strip()
                answer = answer.strip()


                cursor.execute("""
                    INSERT INTO questions (question, answer, is_approved, source, model_quality_score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (question) 
                    DO UPDATE SET 
                        answer = EXCLUDED.answer, 
                        is_approved = TRUE,
                        model_quality_score = 100;
                """, (question, answer, True, "manual_json", 100))
                
                added_count += 1

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ İşlem Tamam: Toplam {added_count} soru veritabanında doğrulandı/güncellendi.")
    
    except Exception as e:
        print(f"❌ Veritabanı Hatası: {e}")

if __name__ == "__main__":
    insert_questions()