import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio

# Загрузка генеративной модели
MODEL_PATH = "generative_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

def get_reply(query: str, chat_history: str = "") -> str:
    prompt = chat_history + "\nUser: " + query + "\nBot:"
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,   # запрещаем повторение n-грамм
        repetition_penalty=1.2      # штраф за повторения
    )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = output[len(prompt):].strip()
    reply = reply.split("\nUser:")[0].strip()
    return reply

# Асинхронная обёртка для инференса
async def get_reply_async(query: str, chat_history: str = "") -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_reply, query, chat_history)

if __name__ == "__main__":
    import asyncio
    query = input("Введите ваш вопрос: ")
    reply = asyncio.run(get_reply_async(query))
    print("Ответ:", reply)
