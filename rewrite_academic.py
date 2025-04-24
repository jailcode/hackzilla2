from openai import AzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

def rewrite_to_academic_tone(text):
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You rewrite text in a passive, academic tone."},
            {"role": "user", "content": f"Rewrite in academic tone: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()
