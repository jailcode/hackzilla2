from openai import AzureOpenAI
from config import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

def paraphrase_text(text):
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who paraphrases text."},
            {"role": "user", "content": f"Paraphrase the following text: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()
