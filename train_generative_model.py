import os
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from tqdm.auto import tqdm

os.makedirs("generative_model", exist_ok=True)

def load_data():
    df = pd.read_csv("data/dialogues_clean.csv", encoding="cp1251")
    # Удаляем строки с пустыми значениями в столбце 'dialogue_clean' и приводим оставшиеся к строке
    df = df.dropna(subset=["dialogue_clean"])
    dialogues = df["dialogue_clean"].astype(str).tolist()
    return dialogues

# Кастомный callback для отображения progress bar с помощью tqdm
class TQDMProgressCallback(TrainerCallback):
    def __init__(self):
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps if state.max_steps is not None else 0
        self.progress_bar = tqdm(total=total_steps, desc="Training progress", leave=True)

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=4, help="Размер батча")
    args = parser.parse_args()

    dialogues = load_data()
    # Создаем единый текст с разделением диалогов, можно добавить специальный токен [SEP] между репликами
    text_data = "\n".join(dialogues)
    data = {"text": [text_data]}
    dataset = Dataset.from_dict(data)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # Устанавливаем токен паддинга, если он отсутствует
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # Если изменили размер словаря (добавили токен), можно вызвать:
    model.resize_token_embeddings(len(tokenizer))

    # Токенизируем текст
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="generative_model",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        disable_tqdm=True  # Отключаем встроенный tqdm, чтобы использовать наш callback
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[TQDMProgressCallback()]
    )

    trainer.train()
    model.save_pretrained("generative_model")
    tokenizer.save_pretrained("generative_model")
    print("Генеративная модель дообучена и сохранена в папке generative_model/")

if __name__ == "__main__":
    main()
