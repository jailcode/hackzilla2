import os
import tempfile
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

BING_API_KEY = AZURE_API_KEY  # Same as Azure for now

def search_sources(query, count=5):
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
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

def citation_agent(query):
    raw_urls = search_sources(query)
    academic_urls = filter_academic_sources(raw_urls)
    return format_as_apa(academic_urls)

citation_tool = Tool(
    name="Citation Generator",
    func=citation_agent,
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
    temp_py = os.path.join(tempfile.gettempdir(), "generated_plot.py")
    temp_png = os.path.join(tempfile.gettempdir(), "output_plot.png")
    code += f"\nplt.savefig('{temp_png}')\nplt.show()"
    with open(temp_py, "w") as f:
        f.write(code)
    try:
        exec_globals = {}
        with open(temp_py, "r") as f:
            exec(f.read(), exec_globals)
        return f"📊 Graph generated and displayed successfully!"
    except Exception as e:
        return f"⚠️ Error running the generated code: {e}"

visualization_tool = Tool(
    name="Visualization Assistant",
    func=visualize_agent,
    description="Generate Python code to visualize data using matplotlib/seaborn."
)

def universal_file_handler(file_path: str) -> str:
    try:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text if text.strip() else "❌ Couldn't extract text from PDF."
        else:
            return f"⚠️ Unsupported file type: {ext}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

universal_file_tool = Tool(
    name="Universal File Handler",
    func=universal_file_handler,
    description="Process PDFs. Input should be the full file path."
)

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
        code = response.choices[0].message.content
        exec(code, globals())
        lines = code.strip().split("\n")
        fn_name = lines[0].split()[1].split("(")[0]
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

tools = [citation_tool, visualization_tool, universal_file_tool]
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
