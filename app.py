from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
from inference import get_reply_async

app = FastAPI(title="Generative Chat Bot API")

# Модель запроса: сообщение пользователя и (опционально) история диалога
class ChatRequest(BaseModel):
    query: str
    chat_history: str = ""  # опционально

# Модель ответа: сгенерированный ответ и обновлённая история диалога
class ChatResponse(BaseModel):
    reply: str
    updated_history: str

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Chat Bot Interface</title>
      <meta charset="utf-8">
      <style>
        body { font-family: Arial, sans-serif; }
        #chat-container { border:1px solid #ccc; height:300px; overflow:auto; padding:10px; }
        #user-input { width:80%; padding:5px; }
        button { padding:5px 10px; }
      </style>
    </head>
    <body>
      <h1>Chat Bot Interface</h1>
      <div id="chat-container"></div>
      <br>
      <input type="text" id="user-input" placeholder="Enter your message">
      <button onclick="sendMessage()">Send</button>
      
      <script>
        let chatHistory = "";
        async function sendMessage() {
          const inputElem = document.getElementById("user-input");
          const message = inputElem.value.trim();
          if (message === "") return;
          inputElem.value = "";
          
          const chatContainer = document.getElementById("chat-container");
          chatContainer.innerHTML += "<p><b>User:</b> " + message + "</p>";
          chatHistory += "\\nUser: " + message;
          
          const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: message, chat_history: chatHistory })
          });
          const data = await response.json();
          chatContainer.innerHTML += "<p><b>Bot:</b> " + data.reply + "</p>";
          chatHistory += "\\nBot: " + data.reply;
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        reply = await get_reply_async(request.query, request.chat_history)
        new_history = request.chat_history + f"\nUser: {request.query}\nBot: {reply}"
        return ChatResponse(reply=reply, updated_history=new_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
