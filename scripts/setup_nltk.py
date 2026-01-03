#!/usr/bin/env python3
"""
Setup script to download required NLTK data
"""
import nltk
import sys

def setup_nltk_data():
    """Download required NLTK data"""
    print("[*] NLTK veri paketleri indiriliyor...")
    
    required_data = ['punkt_tab', 'punkt']
    
    for data_name in required_data:
        try:
            print(f"[*] '{data_name}' indiriliyor...")
            nltk.download(data_name, quiet=False)
            print(f"✅ '{data_name}' başarıyla indirildi.")
        except Exception as e:
            print(f"⚠ '{data_name}' indirilemedi: {e}")
            # Continue with next package
    
    print("✅ NLTK veri paketleri kurulumu tamamlandı.")

if __name__ == "__main__":
    setup_nltk_data()

