from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list stores the full conversation history
conversation = []

print("Chat started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add user's message to history
    conversation.append({"role": "user", "content": user_input})

    # Send the WHOLE conversation so far, not just the latest message
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation
    )

    ai_reply = response.choices[0].message.content
    print(f"AI: {ai_reply}\n")

    # Add AI's reply to history too, so it remembers what it said
    conversation.append({"role": "assistant", "content": ai_reply})