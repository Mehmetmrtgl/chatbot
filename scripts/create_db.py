import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Bağlantı Ayarları
DB_NAME = "hu_chatbot2"
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "123456"
}

def create_everything():
    conn = None
    try:
        # ---------------------------------------------------------
        # 1. ADIM: Varsayılan 'postgres' veritabanına bağlan
        # ---------------------------------------------------------
        print("[*] Sistem başlatılıyor... Postgres sunucusuna bağlanılıyor...")
        conn = psycopg2.connect(dbname="postgres", **DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # CREATE DATABASE işlemi transaksiyon bloğunda çalışmaz
        cursor = conn.cursor()

        # ---------------------------------------------------------
        # 2. ADIM: Veritabanı varlık kontrolü ve oluşturma
        # ---------------------------------------------------------
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"[*] '{DB_NAME}' veritabanı tespit edilemedi. Oluşturma işlemi başlatılıyor...")
            
            # KRİTİK DÜZELTME: SQL komutunu sql.SQL ve sql.Identifier ile formatlıyoruz
            create_db_query = sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(DB_NAME)
            )
            cursor.execute(create_db_query)
            
            print(f"✅ '{DB_NAME}' veritabanı başarıyla oluşturuldu.")
        else:
            print(f"[*] '{DB_NAME}' veritabanı zaten mevcut. Tablo kurulumuna geçiliyor.")

        cursor.close()
        conn.close()

        # ---------------------------------------------------------
        # 3. ADIM: Yeni yaratılan veritabanına bağlan ve şemayı kur
        # ---------------------------------------------------------
        print(f"[*] '{DB_NAME}' veritabanına bağlantı kuruluyor...")
        conn = psycopg2.connect(dbname=DB_NAME, **DB_CONFIG)
        cursor = conn.cursor()

        # Enable pg_trgm extension for similarity() function
        print("[*] PostgreSQL uzantıları kontrol ediliyor...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'
            );
        """)
        extension_exists = cursor.fetchone()[0]
        
        if not extension_exists:
            print("[*] 'pg_trgm' uzantısı etkinleştiriliyor...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                conn.commit()
                print("✅ 'pg_trgm' uzantısı başarıyla etkinleştirildi.")
            except Exception as e:
                print(f"⚠ 'pg_trgm' uzantısı etkinleştirilemedi: {e}")
                print("   Bu uzantı 'similarity()' fonksiyonu için gereklidir.")
        else:
            print("✅ 'pg_trgm' uzantısı zaten etkin.")

        # Check if tables exist
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'questions'
            );
        """)
        tables_exist = cursor.fetchone()[0]

        if not tables_exist:
            # Tabloları sıfırdan kur (Schema Initialization)
            print("[*] Tablolar oluşturuluyor...")
            cursor.execute("""
                CREATE TABLE questions (
                    id SERIAL PRIMARY KEY,
                    question TEXT UNIQUE NOT NULL,
                    answer TEXT,
                    is_approved BOOLEAN DEFAULT FALSE,
                    source TEXT DEFAULT 'manual',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_quality_score INTEGER
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
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            print("✅ Tablolar başarıyla oluşturuldu.")
        else:
            # Tablolar mevcut, eksik sütunları ekle (Migration)
            print("[*] Tablolar mevcut. Eksik sütunlar kontrol ediliyor...")
            
            # Check and add model_quality_score to questions table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='questions' AND column_name='model_quality_score';
            """)
            if not cursor.fetchone():
                print("[*] 'model_quality_score' sütunu questions tablosuna ekleniyor...")
                cursor.execute("ALTER TABLE questions ADD COLUMN model_quality_score INTEGER;")
                print("✅ 'model_quality_score' sütunu eklendi.")
            
            # Check and add session_id to feedback table
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='feedback' AND column_name='session_id';
            """)
            if not cursor.fetchone():
                print("[*] 'session_id' sütunu feedback tablosuna ekleniyor...")
                cursor.execute("ALTER TABLE feedback ADD COLUMN session_id TEXT;")
                print("✅ 'session_id' sütunu eklendi.")
            
            print("✅ Veritabanı şeması güncellendi.")

        conn.commit()
        print("✅ Veritabanı şeması ve tablolar başarıyla yapılandırıldı.")

    except Exception as e:
        print(f"❌ Kritik Hata: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("[*] Bağlantı sonlandırıldı.")

if __name__ == "__main__":
    create_everything()
