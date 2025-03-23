import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Если стоп-слов еще нет — загрузим их
nltk.download('stopwords')

# Создаем директорию для данных, если её нет
os.makedirs("data", exist_ok=True)

def clean_text(text: str) -> str:
    # Удаление всего содержимого в квадратных скобках (включая сами скобки)
    text = re.sub(r'\[.*?\]', ' ', text)  # Важно: это должно быть первым шагом
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление HTML-тегов, цифр и пунктуации
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[\d]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def main():
    # Загружаем исходный датасет из season1.csv
    df = pd.read_csv(r"C:\Users\Administrator\Documents\Projects\Generation_tasks_NLP\data\season1.csv", encoding="cp1251")    
    df = df[df["name"] == "House"]  
    df["dialogue_clean"] = df["line"].apply(clean_text)

    # Сохранение очищенного датасета
    df.to_csv("data/dialogues_clean.csv", index=False, encoding="cp1251")
    print("Датасет успешно очищен и сохранён в data/dialogues_clean.csv")

if __name__ == "__main__":
    main()
