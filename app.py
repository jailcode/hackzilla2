from flask import Flask, request, jsonify, render_template
import os
import tempfile
import re
from fpdf import FPDF
from PyPDF2 import PdfReader
import requests
import openai
from openai import AzureOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import Tool
from langchain.schema import LLMResult
from langchain.llms.base import LLM
from typing import List, Optional
import matplotlib.pyplot as plt


# === Azure OpenAI Settings ===
AZURE_ENDPOINT = "https://ai-subadmin1279ai377683628564.openai.azure.com/"
AZURE_API_KEY = "7IKC9y1hwAM3D1i39HpOGoY8SR8pWXPGg5ryodrsRbJpQrxSsmvGJQQJ99BDACfhMk5XJ3w3AAAAACOGDJX0"
AZURE_DEPLOYMENT = "gpt-4o-mini"
AZURE_API_VERSION = "2024-12-01-preview"

openai.api_type = "azure"
openai.api_base = AZURE_ENDPOINT
openai.api_key = AZURE_API_KEY
openai.api_version = AZURE_API_VERSION

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

class AzureOpenAILLM(LLM):
    model: str = AZURE_DEPLOYMENT
    temperature: float = 0.5

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "azure-openai"

def search_sources(query, count=5):
    headers = {"Ocp-Apim-Subscription-Key": AZURE_API_KEY}
    params = {"q": query, "count": count}
    response = requests.get("https://api.bing.microsoft.com/v7.0/search", headers=headers, params=params)
    results = response.json()
    return [web['url'] for web in results.get('webPages', {}).get('value', [])]

def filter_academic_sources(urls):
    academic_domains = ['.edu', 'springer.com', 'jstor.org', 'researchgate.net']
    return [url for url in urls if any(domain in url for domain in academic_domains)]

def format_as_apa(urls):
    if not urls:
        return "No academic sources found."
    prompt = "Format the following links into APA-style citations:\n" + "\n".join(urls)
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error formatting citations: {e}"

def visualize_agent(query):
    prompt = f"""
    Write Python code using matplotlib or seaborn that visualizes the following data or concept:
    "{query}"
    Only output Python code. Use dummy data if needed.
    """
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.5
    )
    return response.choices[0].message.content

def visualize_agent(query):
    # Generate the Python code for visualizing data
    code = visualize_agent(query)

    # Save the Python code to a temporary file
    temp_py = os.path.join(tempfile.gettempdir(), "generated_plot.py")
    temp_png = os.path.join(tempfile.gettempdir(), "output_plot.png")

    # Append code to save the plot as a PNG file and also show it
    code += f"\nplt.savefig('{temp_png}')\nplt.show()"

    # Write the generated code to a file
    with open(temp_py, "w") as f:
        f.write(code)

    try:
        # Execute the code to generate and display the plot
        exec_globals = {}
        with open(temp_py, "r") as f:
            exec(f.read(), exec_globals)
        return f"📊 Graph generated and displayed successfully! Saved at {temp_png}"
    except Exception as e:
        return f"⚠️ Error running the generated code: {e}"

def audio_to_text(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        return transcript.get("text", "")
    except Exception as e:
        return f"Transcription failed: {str(e)}"

# Tools and helper functions
def universal_file_handler(file_path: str) -> str:
    try:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            reader = PdfReader(file_path)
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif ext in ['.mp3', '.wav', '.m4a', '.aac']:
            return audio_to_text(file_path)
        return f"⚠️ Unsupported file type: {ext}"
    except Exception as e:
        return f"❌ Error: {str(e)}"
# Define tools for the AI agent
tools = [
    Tool(
        name="Citation Generator",
        func=lambda query: format_as_apa(filter_academic_sources(search_sources(query))),
        description="Use this to find and format citations from academic web sources. Input should be a research query."
    ),
    Tool(
        name="Visualization Assistant",
        func=visualize_agent,
        description="Generate Python code to visualize data using matplotlib/seaborn."
    ),
    Tool(
        name="Audio To Text",
        func=audio_to_text,
        description="Convert an audio file to text using Azure OpenAI Whisper. Input should be the full path to the audio file."
    ),
    Tool(
        name="Universal File Handler",
        func=universal_file_handler,
        description="Process PDFs and audio files. Input should be the full file path."
    )
]
def save_text_as_pdf(text: str, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(output_path)

def run_copilot(user_query=None):
    if user_query:
        # Process the single query
        agent = initialize_agent(
            tools=tools,  # Use the defined tools
            llm=AzureOpenAILLM(),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        try:
            result = agent.invoke(user_query)
            return result
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize Flask app
app = Flask(__name__)

# Route to serve the homepage
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # This will look inside the templates/ folder

# Route to process user queries
@app.route("/process-query", methods=["POST"])
def process_query():
    user_query = request.form.get("userQuery")  # Get the input from the form
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Pass the query to the `run_copilot` logic
    result = run_copilot(user_query)

    # Return the result as a JSON response
    return jsonify({"result": result})

# Route to handle file uploads and processing
@app.route("/upload-file", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    # Process the file
    result = universal_file_handler(temp_path)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)