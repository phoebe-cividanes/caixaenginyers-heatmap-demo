import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def ask_gemini(system_prompt, user_question, model_name="gemini-2.5-flash"):
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt  
        ),
        contents=user_question
    )
    return response.text

if __name__ == "__main__":
    system_prompt = "You are a helpful data analyst. Answer questions only using the provided data, and explain your reasoning clearly."
    user_question = "Based on this data: 'Customer sales for 2025: Q1 - $10,000, Q2 - $12,500, Q3 - $15,000, Q4 - $13,000. What is the annual total and which quarter saw the highest sales?'"
    
    answer = ask_gemini(system_prompt, user_question)
    print("Gemini's answer:", answer)

