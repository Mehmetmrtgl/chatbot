# app/db/config.py

import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="hu_chatbot2",         # kendi veritabanı adın
        user="postgres",          # kullanıcı adın
        password="123456"         # şifren
    )
