import json
import psycopg2

DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "hu_chatbot2",
    "user": "postgres",
    "password": "123456"
}
JSON_PATH = "../data/manual_questions.json"

def insert_questions():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    for item in data:
        question = item.get("question")
        answer = item.get("answer")

        cursor.execute("""
            INSERT INTO questions (question, answer, is_approved, source)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (question) DO NOTHING;
        """, (question, answer, True, "manual"))

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Manuel sorular başarıyla eklendi.")

if __name__ == "__main__":
    insert_questions()
