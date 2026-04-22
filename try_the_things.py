import os

from groq import Groq
from dotenv import load_dotenv

# Lädt die Variablen aus der .env-Datei in die Systemumgebung
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)