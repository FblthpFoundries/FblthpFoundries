import openai
from constants import API_KEY
# Replace 'your-api-key-here' with your actual OpenAI API key
openai.api_key = API_KEY

def test_openai_connection():
    response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    #    {"role": "system", "content": header},
        {"role": "user", "content": "Hey there!"}
    ],
    )
    print(response)

if __name__ == "__main__":
    test_openai_connection()