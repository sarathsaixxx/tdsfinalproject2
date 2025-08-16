import os
import json
import google.generativeai as genai

# Get the API key from environment variable
api_key = "AIzaSyByTFQszk0KXL9GR8y7lLMAivDLZwCxZiY"

if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = f"""
You are given a set of questions and a metadata description of available data.

Questions:
some question

Metadata:
metadata with links

Your job:
1. Write Python code that answers the questions using the data described above.
2. Save **all answers** as structured JSON to this file: "result.json".
3. If a question produces a chart / image (like a plot / network graph), convert it to **base64 PNG** & include it at JSON like:
   "my_chart": "<base64 PNG>"

Instructions:
- Do NOT return explanation or markdown — just code and library list.
- Use only standard libraries + pandas, matplotlib, networkx, seaborn, numpy, etc.
- Any images should be saved in base64 and embedded in result.json, not as separate files.
- Output MUST be a JSON object with keys and values relevant to the question.

Final output format:
{{
  "code": "your Python code as string",
  "libraries": ["required", "libraries"]
}}

Make sure your code:
- Runs successfully in one go.
- Writes all outputs to result.json as instructed.
"""

async def parse_question_with_llm(question_text, uploaded_files=None, urls=None, folder="uploads"):
    uploaded_files = uploaded_files or []
    urls = urls or []

    user_prompt = f"""
Question:
"{question_text}"

Uploaded files:
"{uploaded_files}"

URLs:
"{urls}"

You are a data extraction specialist.
Your task is to generate Python 3 code that loads, scrapes, or reads the data needed to answer the user's question.

1(a). Always store the final dataset in a file as {folder}/data.csv file. And if you need to store other files then also store them in this folder. Lastly, add the path and a brief description about the file in "{folder}/metadata.txt".
1(b). Create code to collect metadata about the data that you collected from scraping (eg. storing details of df using df.info, df.columns, df.head() etc.) in a "{folder}/metadata.txt" file that will help other model to generate code. Add code for creating any folder that doesn't exist like "{folder}".

2. Do not perform any analysis or answer the question. Only write code to collect or add metadata.

3. The code must be self-contained and runnable without manual edits.

4. Use only Python standard libraries plus pandas, numpy, beautifulsoup4, and requests unless otherwise necessary.

5. If the data source is a webpage, download and parse it. If it’s a CSV/Excel, read it directly.

6. Do not explain the code.

7. Output only valid Python code.

8. Just scrap the data don’t do anything fancy.

Return a JSON with:
1. The 'code' field — Python code that answers the question.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that came installed with python like io.
4. Your output will be executed inside a Python REPL.
5. Don't add comments

Only return JSON like:
{{
  "code": "<...>",
  "libraries": ["pandas", "matplotlib"],
  "questions": ["..."]
}}

lastly i am saying again don't try to solve these questions.
in metadata also add JSON answer format if present.
"""

    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        [SYSTEM_PROMPT, user_prompt],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    # Path to the file
    file_path = os.path.join(folder, "metadata.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
    
    return json.loads(response.text)

SYSTEM_PROMPT2 = """
You are a data analysis assistant.  
Your job is to:
1. Write Python code to solve these questions with provided metadata.
2. List all Python libraries that need to be installed for the code to run.
3. Also add code to save the result to "{folder}/result.json" or any filetype you find suitable (eg. save img files like "{folder}/img.png").

Do not include explanations, comments, or extra text outside the JSON.
"""

async def answer_with_data(question_text, folder="uploads"):
    metadata_path = os.path.join(folder, "metadata.txt")
    with open(metadata_path, "r") as file:
        metadata = file.read()

    user_prompt = f"""
Question:
{question_text}

metadata:
{metadata}

Return a JSON with:
1. The 'code' field — Python code that answers the question.
2. The 'libraries' field — list of required pip install packages.
3. Don't add libraries that came installed with python like "io".
4. Your output will be executed inside a Python REPL.
5. Don't add comments
6. Convert any image/visualisation if present, into base64 PNG and add it to the result.

You must respond **only** in valid JSON with these properties:

  "code": "string — Python scraping code as plain text",
  "libraries": ["string — names of required libraries"]

lastly follow answer format and save answer of questions in result as JSON file.
"""

    # Path to the file
    file_path = os.path.join(folder, "result.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    model = genai.GenerativeModel(MODEL_NAME)

    # SYSTEM_PROMPT2 needs to be formatted with the folder
    system_prompt2 = SYSTEM_PROMPT2.format(folder=folder)

    response = model.generate_content(
        [system_prompt2, user_prompt],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)
