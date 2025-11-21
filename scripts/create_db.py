import psycopg2

DB_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "dbname": "hu_chatbot2",
    "user": "postgres",
    "password": "123456"
}

def create_questions_table():
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    cursor.execute("""
        DROP TABLE IF EXISTS feedback CASCADE;
        DROP TABLE IF EXISTS questions CASCADE;

        CREATE TABLE questions (
            id SERIAL PRIMARY KEY,
            question TEXT UNIQUE NOT NULL,
            answer TEXT,
            is_approved BOOLEAN DEFAULT FALSE,
            source TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE chat_logs (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT CHECK (role IN ('user', 'assistant', 'system')) NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE feedback (
            id SERIAL PRIMARY KEY,
            question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
            feedback_type TEXT CHECK (feedback_type IN ('like', 'dislike')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 'questions' tablosu başarıyla oluşturuldu.")

if __name__ == "__main__":
    create_questions_table()
