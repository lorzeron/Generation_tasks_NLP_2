import time
from inference import get_reply

queries = ["болит нога", "почему ты такой грубый?", "это волчанка?"]

# Тест скорости
start = time.time()
for q in queries:
    get_reply(q)
print(f"Время на 3 запроса: {time.time() - start:.4f} сек")



#Время на 3 запроса: 0.0040 сек