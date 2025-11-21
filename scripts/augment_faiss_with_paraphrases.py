from transformers import pipeline
from scripts.faiss_utils import add_to_faiss_index

paraphraser = pipeline("text2text-generation", model="Vamsi/T5-paraphrase-paws")

def generate_paraphrases(question):
    input_text = f"paraphrase: {question} </s>"
    outputs = paraphraser(input_text, max_length=64, num_return_sequences=3, do_sample=True)
    return [o["generated_text"] for o in outputs]

from scripts.db_utils import get_all_questions

all_questions = get_all_questions()
for q in all_questions:
    paraphrases = generate_paraphrases(q)
    for p in paraphrases:
        add_to_faiss_index(p)
