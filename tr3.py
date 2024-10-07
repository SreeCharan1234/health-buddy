import requests
import json
def get_gemini_response(prompt):
    api_key = "AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE"
    url = "https://api.geminiai.com/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Could not get response from Gemini."

# Generate initial question
question = get_gemini_response("Generate a question related to health and well-being.")

# Collect user responses
responses = []
for i in range(10):
    print(question)
    response = input("> ")
    responses.append(response)

    # Generate next question based on previous responses
    prompt = f"Generate a follow-up question based on the previous responses: {responses}"
    question = get_gemini_response(prompt)

# Hypothetical "diagnosis" based on responses (for educational purposes only)
if "headaches" in responses and "vision problems" in responses:
    print("Based on your responses, you might want to consult a doctor about potential migraines.")
else:
    print("Your responses don't indicate any immediate concerns, but it's always a good idea to consult a healthcare professional if you have any persistent symptoms.")