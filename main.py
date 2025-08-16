from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging

from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper funtion to show last 25 words of string s
def last_n_words(s, n=25):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0



@app.post("/api")
async def analyze(request: Request):
    # Create a unique folder for this request
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    # Setup logging for this request
    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    # Remove previous handlers if any (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Also log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Step-1: Folder created: %s", request_folder)

    form = await request.form()
    question_text = None
    saved_files = {}

    # Save all uploaded files to the request folder
    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:  # It's a file
            file_path = os.path.join(request_folder, value.filename)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await value.read())
            saved_files[field_name] = file_path

            # If it's questions.txt, read its content
            if field_name == "questions.txt":
                async with aiofiles.open(file_path, "r") as f:
                    question_text = await f.read()
        else:
            saved_files[field_name] = value

    # Fallback: If no questions.txt, use the first file as question
    if question_text is None and saved_files:
        first_file = next(iter(saved_files.values()))
        async with aiofiles.open(first_file, "r") as f:
            question_text = await f.read()
    
    logger.info("Step-2: File sent %s", saved_files)


    # ✅ 4. Get code steps from LLM

    max_attempts = 3
    attempt = 0
    response = None
    
    
    while attempt < max_attempts:
        logger.info("Step-3: Getting scrap code and metadata from llm. Tries count = %d", attempt)
        try:
            response = await parse_question_with_llm(
                question_text=question_text,
                uploaded_files=saved_files,
                folder=request_folder
            )
            # Check if response is a valid dict (parsed JSON)
            if isinstance(response, dict):
                break
        except Exception as e:
            question_text = question_text + last_n_words(str(e), 50)
            logger.error("Step-3: Error in parsing the result. %s", last_n_words(question_text))
        attempt += 1


    if not isinstance(response, dict):
        logger.error("Error: Could not get valid response from LLM after retries.")
        return JSONResponse({"message": "Error: Could not get valid response from LLM after retries."})



    logger.info("Step-3: Response from scrapping: %s", last_n_words(response))

    # ✅ 5. Execute generated code safely
    execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)

    logger.info("Step-4: Execution result of the scrape code: %s", last_n_words(execution_result["output"]))

    count = 0
    while execution_result["code"] == 0 and count < 3:
        logger.error("Step-4: Error occured while scrapping. Tries count = %d", count)
        new_question_text = str(question_text) + "previous time this error occured" + last_n_words(str(execution_result["output"]), 50)
        try:
            response = await parse_question_with_llm(
                question_text=new_question_text,
                uploaded_files=saved_files,
                folder=request_folder
            )
        except Exception as e:
            logger.error("Step-4: error occured while reading json. %s", last_n_words(e))

        logger.info("Step-3: Response from scrapping: %s", last_n_words(response))

        execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)

        logger.info("Step-4: Execution result of the scrape code: %s", last_n_words(execution_result["output"]))
        csv_path = os.path.join(request_folder, "data.csv")
        csv_retry_count = 0
        max_csv_retries = 3

        while os.path.exists(csv_path) and is_csv_empty(csv_path) and csv_retry_count < max_csv_retries:
            logger.warning("data.csv is present but empty. Retrying code generation and  execution. Attempt %d", csv_retry_count + 1)
            response = await parse_question_with_llm(
                question_text=new_question_text + str("there is nothing present in data.csv file."),
                uploaded_files=saved_files,
                folder=request_folder
            )
            execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
            csv_retry_count += 1

        count += 1

    if execution_result["code"] == 1:
        execution_result = execution_result["output"]
    else:
        logger.error("error occured while scrapping.")
        return JSONResponse({"message": "error occured while scrapping."})

    

    # 6. get answers from llm
    max_attempts = 3
    attempt = 0
    gpt_ans = None
    response_questions = response["questions"]

    while attempt < max_attempts:
        logger.info("Step-5: Getting execution code from llm. Tries count = %d", attempt)
        try:
            gpt_ans = await answer_with_data(response_questions, folder=request_folder)
            logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
            if isinstance(gpt_ans, dict):
                break
        except Exception as e:
            logger.error("Step-5: Error: %s", e)
            response_questions += last_n_words(str(e), 50)
        attempt += 1
    
    if not isinstance(gpt_ans, dict):
        logger.error("Error: Could not get valid response from answer_with_data after retries.")
        return JSONResponse({"message": "Error: Could not get valid response from answer_with_data after retries."})
    
    # 7. Executing code
    try:
        logger.info("Step-6: Executing final code. Tries count = 0")
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
        logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
    except Exception as e:
        logger.error("Step-6: Trying after it caught under except block-wrong json format. Tries count = 1 %s", last_n_words(e))
        logger.info("Step-5: Getting execution code from llm. Tries count = %d", attempt+1)
        gpt_ans = await answer_with_data(str(response["questions"])+str("Please follow the json structure"), folder=request_folder)
        logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
        logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
        

    count = 0
    json_str = 1
    while final_result["code"] == 0 and count < 3:
        logger.error("Step-6: Error occured while executing code. Tries count = %d", count+2)
        new_question_text = str(response["questions"]) + "previous time this error occured" + last_n_words(str(final_result["output"]), 50)
        if json_str == 0:
            new_question_text += "follow the structure {'code': '', 'libraries': ''}"
        logger.info("Step-5: Getting execution code from llm. Tries count = %d", attempt+1)
        try:
            gpt_ans = await answer_with_data(new_question_text, folder=request_folder)
            logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
        except Exception as e:
            logger.error("Step-5: Json parsing error. %s", last_n_words(e))
        try:
            json_str = 0
            final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
            logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
            json_str = 1
        except Exception as e:
            logger.error("Exception occurred: %s", e)
            count -= 1

        count += 1

    if final_result["code"] == 1:
        final_result = final_result["output"]
    else:
        result_path = os.path.join(request_folder, "result.json")
        with open(result_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)

    result_path = os.path.join(request_folder, "result.json")
    with open(result_path, "r") as f:
        try:
            data = json.load(f)
            logger.info("Step-7: send result back")
            return JSONResponse(content=data)
        except Exception as e:
            logger.error("Step-7: Error occur while sending result: %s", last_n_words(e))
            # Return raw content if JSON parsing fails
            f.seek(0)
            raw_content = f.read()
            return JSONResponse({"message": f"Error occured while processing result.json: {e}", "raw_result": raw_content})