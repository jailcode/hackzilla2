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

citation_tool = Tool(
    name="Citation Generator",
    func=lambda query: format_as_apa(filter_academic_sources(search_sources(query))),
    description="Use this to find and format citations from academic web sources. Input should be a research query."
)

def generate_visualization_code(query):
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
    code = generate_visualization_code(query)
    code += f"\nplt.savefig('{tempfile.gettempdir()}/output_plot.png')\nplt.show()"

    try:
        exec(code, {})
        return "📊 Graph generated and displayed successfully!"
    except Exception as e:
        return f"⚠️ Error running the generated code: {e}"

visualization_tool = Tool(
    name="Visualization Assistant",
    func=visualize_agent,
    description="Generate Python code to visualize data using matplotlib/seaborn."
)

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

audio_tool = Tool(
    name="Audio To Text",
    func=audio_to_text,
    description="Convert an audio file to text using Azure OpenAI Whisper. Input should be the full path to the audio file."
)

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

universal_file_tool = Tool(
    name="Universal File Handler",
    func=universal_file_handler,
    description="Process PDFs and audio files. Input should be the full file path."
)

def clean_code_block(code: str) -> str:
    cleaned = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()

def create_tool_dynamically(user_query):
    generation_prompt = f"""
    You are an AI assistant. The user wants help with this task:
    "{user_query}"
    Write a Python function using OpenAI's API that helps complete this task. Output only the function code.
    """
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": generation_prompt}],
            max_tokens=4096,
            temperature=0.7
        )
        code = clean_code_block(response.choices[0].message.content)

        exec(code, globals())
        match = re.search(r"def (\w+)\(", code)
        if not match:
            raise ValueError("Couldn't find function definition in generated code.")
        fn_name = match.group(1)

        new_tool = Tool(
            name=fn_name.replace("_", " ").title(),
            func=globals()[fn_name],
            description=f"Generated tool to handle: {user_query}"
        )
        tools.append(new_tool)
        print(f"✅ New agent '{fn_name}' created.")
        return new_tool
    except Exception as e:
        print(f"❌ Failed to create tool: {e}")
        return None

tools = [citation_tool, visualization_tool, audio_tool, universal_file_tool]
llm = AzureOpenAILLM()

print("\n🎓 Welcome to Student Copilot with Meta-Agent Generator!")

def save_text_as_pdf(text: str, output_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(output_path)

while True:
    user_query = input("\n🧑‍💻 What do you need help with? (type 'exit' to quit)\n> ")
    if user_query.lower() in ['exit', 'quit']:
        print("👋 Goodbye!")
        break

    if os.path.isfile(user_query.strip()):
        print("📂 Detected file. Processing...")
        result = universal_file_handler(user_query.strip())
        print("\n📤 Output:\n", result)

        to_pdf = input("\n💾 Would you like to save the result as a PDF? (yes/no)\n> ").strip().lower()
        if to_pdf in ['yes', 'y']:
            output_pdf = os.path.splitext(user_query.strip())[0] + "_output.pdf"
            save_text_as_pdf(result, output_pdf)
            print(f"✅ PDF saved at: {output_pdf}")
        continue

    if os.path.isdir(user_query.strip()):
        files = os.listdir(user_query.strip())
        audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a', '.aac'))]

        if not audio_files:
            print("⚠️ No audio files found in that directory.")
            continue

        print("🎧 Found these audio files:")
        for idx, fname in enumerate(audio_files):
            print(f"[{idx}] {fname}")

        try:
            choice = int(input("Select a file number to transcribe:\n> "))
            selected_file = os.path.join(user_query.strip(), audio_files[choice])
            transcription = audio_tool.func(selected_file)
            print("\n📝 Transcribed Text:\n", transcription)

            to_pdf = input("\n📄 Save as PDF? (yes/no)\n> ").strip().lower()
            if to_pdf in ['yes', 'y']:
                pdf_path = os.path.splitext(selected_file)[0] + ".pdf"
                save_text_as_pdf(transcription, pdf_path)
                print(f"✅ PDF saved at: {pdf_path}")
        except (IndexError, ValueError):
            print("❌ Invalid selection.")
        continue

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    try:
        result = agent.invoke(user_query)
        print("\n🤖 Copilot says:\n", result)
    except Exception as e:
        print("🤖 Couldn't handle that. Trying to create a new agent...")
        tool = create_tool_dynamically(user_query)
        if tool:
            print("🔁 Retrying with new agent...")
            agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent.agent, tools=tools, verbose=True)
            print("\n🤖 Copilot says:\n", agent_executor.invoke(user_query))
        else:
            print("🙁 No tool could be created. Try again.")
    