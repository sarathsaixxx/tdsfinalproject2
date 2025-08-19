import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI
from dotenv import load_dotenv

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# LangChain / LLM imports (keep as you used)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Serve ui.html at /test
from fastapi.responses import HTMLResponse
@app.get("/test", response_class=HTMLResponse)
async def serve_ui():
    try:
        with open("ui.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure ui.html is in the same directory as app.py</p>", status_code=404)
        
# -------------------- Robust Gemini LLM with fallback --------------------
from collections import defaultdict
import time
from langchain_google_genai import ChatGoogleGenerativeAI

# Config
GEMINI_KEYS = ['AIzaSyC6pEDxqNHybznG78kofVEHWAPMMdSMzH0','AIzaSyDrI0iMJNTwQCLKvsc-Hlf039DpDHe4Sbg','AIzaSyAH9gDsubojeFXfkKQklpEaoTc3KSDOJMk','AIzaSyAh7bIF8psaOGo5nycfChTS9TT1mB1X9co','AIzaSyAHI5zbY8UzKkBcxjECqflWCda677z66Ic','AIzaSyBkawl72cy-H4uJtJ7Rzwq8fK0t01gZAjg','AIzaSyDu55eFkyA4kOtw2MC0Zg9UVQKQHoIdx0U','AIzaSyDcHXn6xS6IZgc--BGTfIInRr_gcbwgXTY','AIzaSyCvjpNLPU-Mwtjrn5xrIL_BXhE20YDTVsg','AIzaSyDe_ZBU2pHysApuRUlGTPKT8VMBN7oTf6w']
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

MAX_RETRIES_PER_KEY = 3
TIMEOUT = 45
BACKOFF_BASE = 1.0  # Base delay in seconds
MAX_BACKOFF = 30.0  # Maximum delay in seconds

# Error patterns that indicate specific API issues
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "429", "too many requests"]
INTERNAL_ERROR_KEYWORDS = ["500", "internal server error", "internal error"]
SERVICE_UNAVAILABLE_KEYWORDS = ["503", "service unavailable", "temporarily unavailable"]
PERMANENT_ERROR_KEYWORDS = ["401", "unauthorized", "invalid api key", "api key not found"]

if not GEMINI_KEYS:
    logger.error("No Gemini API keys found. Please set them in your environment as gemini_api_1, gemini_api_2, etc.")
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")
else:
    logger.info(f"Loaded {len(GEMINI_KEYS)} Gemini API keys")
    logger.info(f"Using model hierarchy: {MODEL_HIERARCHY}")

# -------------------- LLM wrapper --------------------

class LLMWithFallback:
    def __init__(self, keys=None, models=None, temperature=0, backoff_base=BACKOFF_BASE, max_retries=MAX_RETRIES_PER_KEY):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature

        # Enhanced logging and tracking
        self.key_failure_count = defaultdict(int)      # Total failures per key
        self.key_success_count = defaultdict(int)      # Successful calls per key
        self.key_last_used = defaultdict(float)        # Last usage timestamp per key
        self.key_cooldown_until = defaultdict(float)   # Cooldown period per key
        self.permanently_bad_keys = set()              # Blacklisted keys
        self.model_key_failures = defaultdict(lambda: defaultdict(int))  # Track model+key combo failures
        
        self.current_llm = None
        self.backoff_base = backoff_base
        self.max_retries = max_retries
        self.key_rotation_index = 0  # For round-robin key selection
        
        logger.info(f"Initialized LLM with {len(self.keys)} keys and {len(self.models)} models")

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error to determine appropriate handling strategy"""
        error_lower = str(error_msg).lower()
        
        if any(keyword in error_lower for keyword in PERMANENT_ERROR_KEYWORDS):
            return "permanent"
        elif any(keyword in error_lower for keyword in QUOTA_KEYWORDS):
            return "quota"  
        elif any(keyword in error_lower for keyword in INTERNAL_ERROR_KEYWORDS):
            return "internal"
        elif any(keyword in error_lower for keyword in SERVICE_UNAVAILABLE_KEYWORDS):
            return "unavailable"
        else:
            return "unknown"

    def _should_skip_key(self, key: str, model: str) -> bool:
        """Check if a key should be skipped due to cooldown or permanent issues"""
        current_time = time.time()
        
        # Skip permanently bad keys
        if key in self.permanently_bad_keys:
            return True
            
        # Skip keys in cooldown period
        if current_time < self.key_cooldown_until[key]:
            return True
            
        # Skip if this model+key combo has failed too many times recently
        if self.model_key_failures[model][key] >= self.max_retries:
            return True
            
        return False

    def _apply_cooldown(self, key: str, error_category: str):
        """Apply appropriate cooldown based on error type"""
        current_time = time.time()
        
        if error_category == "quota":
            # Longer cooldown for quota issues (5-15 minutes)
            cooldown = min(300 + (self.key_failure_count[key] * 60), 900)
        elif error_category in ["internal", "unavailable"]:
            # Medium cooldown for server issues (30 seconds to 5 minutes)
            cooldown = min(30 + (self.key_failure_count[key] * 30), 300)
        elif error_category == "permanent":
            # Permanent ban
            self.permanently_bad_keys.add(key)
            return
        else:
            # Default cooldown for unknown errors
            cooldown = min(10 + (self.key_failure_count[key] * 10), 120)
        
        self.key_cooldown_until[key] = current_time + cooldown
        logger.warning(f"Applied {cooldown}s cooldown to key {key[:8]}... due to {error_category} error")

    def _get_next_key(self, available_keys: List[str]) -> str:
        """Get next key using round-robin with preference for least recently used"""
        if not available_keys:
            return None
            
        # Sort keys by last usage time (least recently used first)
        sorted_keys = sorted(available_keys, key=lambda k: self.key_last_used[k])
        
        # Use round-robin among top candidates
        if len(sorted_keys) > 1:
            self.key_rotation_index = (self.key_rotation_index + 1) % len(sorted_keys)
            return sorted_keys[self.key_rotation_index % len(sorted_keys)]
        else:
            return sorted_keys[0]

    def _get_llm_instance(self):
        """
        Enhanced key/model selection with intelligent fallback and cooldown management
        """
        last_error = None
        total_attempts = 0
        max_total_attempts = len(self.keys) * len(self.models) * 2  # Allow some retries

        for model in self.models:
            logger.info(f"Trying model: {model}")
            
            # Get available keys for this model (not in cooldown)
            available_keys = [k for k in self.keys if not self._should_skip_key(k, model)]
            
            if not available_keys:
                logger.warning(f"No available keys for model {model}, trying next model")
                continue

            # Try each available key for this model
            for attempt in range(min(len(available_keys), 3)):  # Limit attempts per model
                if total_attempts >= max_total_attempts:
                    break
                    
                key = self._get_next_key(available_keys)
                if not key:
                    break
                    
                available_keys.remove(key)  # Remove from current attempt pool
                total_attempts += 1
                
                try:
                    logger.info(f"Attempting with key {key[:8]}... and model {model} (attempt {total_attempts})")
                    
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key,
                        timeout=TIMEOUT
                    )
                    
                    # Test the instance with a simple call
                    test_response = llm_instance.invoke("Hello")
                    
                    # Success! Update tracking
                    self.key_last_used[key] = time.time()
                    self.key_success_count[key] += 1
                    self.model_key_failures[model][key] = 0  # Reset failure count for this combo
                    self.current_llm = llm_instance
                    
                    logger.info(f"Successfully connected with key {key[:8]}... and model {model}")
                    return llm_instance

                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    error_category = self._categorize_error(error_msg)
                    
                    # Update failure tracking
                    self.key_failure_count[key] += 1
                    self.model_key_failures[model][key] += 1
                    
                    logger.error(f"Key {key[:8]}... failed with {model}: {error_category} - {error_msg}")
                    
                    # Apply appropriate cooldown
                    self._apply_cooldown(key, error_category)
                    
                    # Calculate backoff delay
                    delay = min(
                        self.backoff_base * (2 ** (self.key_failure_count[key] - 1)),
                        MAX_BACKOFF
                    )
                    
                    if error_category in ["quota", "unavailable"]:
                        delay *= 2  # Longer delay for these errors
                    
                    logger.info(f"Waiting {delay}s before next attempt...")
                    time.sleep(delay)

        # Log final statistics before failing
        logger.error("All keys/models exhausted. Final statistics:")
        for key in self.keys:
            logger.error(f"Key {key[:8]}...: {self.key_success_count[key]} successes, {self.key_failure_count[key]} failures")

        raise RuntimeError(f"All {len(self.keys)} keys and {len(self.models)} models failed after {total_attempts} attempts. Last error: {last_error}")

    def get_health_status(self) -> dict:
        """Get current health status of all keys"""
        current_time = time.time()
        status = {
            "total_keys": len(self.keys),
            "available_keys": 0,
            "permanently_bad_keys": len(self.permanently_bad_keys),
            "keys_in_cooldown": 0,
            "key_details": {}
        }
        
        for key in self.keys:
            key_id = key[:8] + "..."
            is_available = not self._should_skip_key(key, self.models[0])  # Check with first model
            cooldown_remaining = max(0, self.key_cooldown_until[key] - current_time)
            
            status["key_details"][key_id] = {
                "available": is_available,
                "successes": self.key_success_count[key],
                "failures": self.key_failure_count[key],
                "cooldown_remaining_seconds": int(cooldown_remaining),
                "permanently_bad": key in self.permanently_bad_keys
            }
            
            if is_available:
                status["available_keys"] += 1
            elif cooldown_remaining > 0:
                status["keys_in_cooldown"] += 1
                
        return status

    # Required by LangChain agent
    def bind_tools(self, tools):
        llm_instance = self._get_llm_instance()
        return llm_instance.bind_tools(tools)

    # Keep .invoke interface with retry logic
    def invoke(self, prompt):
        max_invoke_retries = 2
        for retry in range(max_invoke_retries):
            try:
                llm_instance = self._get_llm_instance()
                return llm_instance.invoke(prompt)
            except Exception as e:
                logger.error(f"Invoke attempt {retry + 1} failed: {str(e)}")
                if retry == max_invoke_retries - 1:
                    raise
                time.sleep(2)  # Brief delay before retry




# class LLMWithFallback:
#     def __init__(self, keys=None, models=None, temperature=0):
#         self.keys = keys or GEMINI_KEYS
#         self.models = models or MODEL_HIERARCHY
#         self.temperature = temperature
#         self.slow_keys_log = defaultdict(list)
#         self.failing_keys_log = defaultdict(int)
#         self.current_llm = None  # placeholder for actual ChatGoogleGenerativeAI instance

#     def _get_llm_instance(self):
#         last_error = None
#         for model in self.models:
#             for key in self.keys:
#                 try:
#                     llm_instance = ChatGoogleGenerativeAI(
#                         model=model,
#                         temperature=self.temperature,
#                         google_api_key=key
#                     )
#                     self.current_llm = llm_instance
#                     return llm_instance
#                 except Exception as e:
#                     last_error = e
#                     msg = str(e).lower()
#                     if any(qk in msg for qk in QUOTA_KEYWORDS):
#                         self.slow_keys_log[key].append(model)
#                     self.failing_keys_log[key] += 1
#                     time.sleep(0.5)
#         raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

#     # Required by LangChain agent
#     def bind_tools(self, tools):
#         llm_instance = self._get_llm_instance()
#         return llm_instance.bind_tools(tools)

#     # Keep .invoke interface
#     def invoke(self, prompt):
#         llm_instance = self._get_llm_instance()
#         return llm_instance.invoke(prompt)


LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    Returns:
        keys_list: list of keys in order
        type_map: dict key -> casting function
    """
    import re
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map




# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    # create file content
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    # inject df if a pickle path provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        preamble.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    # ensure results printed as json
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path],
                                   capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            # collect stderr and stdout for debugging
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        # parse stdout as json
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# -----------------------------
# LLM agent setup
# -----------------------------
# llm = ChatGoogleGenerativeAI(
#     model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
#     temperature=0,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
# -------------------- Initialize LLM --------------------
logger.info("Initializing LLM with enhanced fallback system...")
llm = LLMWithFallback(temperature=0)
logger.info("LLM initialization complete")
# -----------------------------

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# Prompt: instruct agent to call the tool and output JSON only
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
# -----------------------------
def run_agent_safely(llm_input: str) -> Dict:
    """
    1. Run the agent_executor.invoke to get LLM output with enhanced error handling
    2. Extract JSON, get 'code' and 'questions'
    3. Detect scrape_url_to_dataframe("...") calls in code, run them here, pickle df and inject before exec
    4. Execute the code in a temp file and return results mapping questions -> answers
    """
    max_agent_retries = 2
    
    for attempt in range(max_agent_retries):
        try:
            logger.info(f"Agent attempt {attempt + 1}/{max_agent_retries}")
            
            # Check LLM health before attempting
            if hasattr(llm, 'get_health_status'):
                health = llm.get_health_status()
                logger.info(f"LLM Health: {health['available_keys']}/{health['total_keys']} keys available")
                
                if health['available_keys'] == 0:
                    logger.warning("No API keys available, waiting for cooldowns...")
                    time.sleep(10)  # Wait a bit for cooldowns
            
            response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            
            if not raw_out:
                if attempt < max_agent_retries - 1:
                    logger.warning(f"Agent returned no output on attempt {attempt + 1}, retrying...")
                    time.sleep(5)
                    continue
                else:
                    return {"error": f"Agent returned no output after {max_agent_retries} attempts. Full response: {response}"}

            parsed = clean_llm_output(raw_out)
            if "error" in parsed:
                if attempt < max_agent_retries - 1:
                    logger.warning(f"Failed to parse agent output on attempt {attempt + 1}: {parsed['error']}")
                    time.sleep(3)
                    continue
                else:
                    return parsed

            if not isinstance(parsed, dict) or "code" not in parsed or "questions" not in parsed:
                if attempt < max_agent_retries - 1:
                    logger.warning(f"Invalid agent response format on attempt {attempt + 1}")
                    time.sleep(3)
                    continue
                else:
                    return {"error": f"Invalid agent response format: {parsed}"}

            code = parsed["code"]
            questions: List[str] = parsed["questions"]

            # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            pickle_path = None
            if urls:
                # For now support only the first URL (agent may code multiple scrapes; you can extend this)
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                # create df and pickle it
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name
                # Make sure agent's code can reference df/data: we will inject the pickle loader in the temp script

            # Execute code in temp python script
            exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
            if exec_result.get("status") != "success":
                return {"error": f"Execution failed: {exec_result.get('message', exec_result)}", "raw": exec_result.get("raw")}

            # exec_result['result'] should be results dict
            results_dict = exec_result.get("result", {})
            # Map to original questions (they asked to use exact question strings)
            output = {}
            for q in questions:
                output[q] = results_dict.get(q, "Answer not found")
            return output

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
            
            # Check if it's an API-related error that we should retry
            if any(keyword in error_msg.lower() for keyword in ["429", "500", "503", "quota", "rate limit"]):
                if attempt < max_agent_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s
                    logger.info(f"API error detected, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            
            if attempt == max_agent_retries - 1:
                logger.exception("run_agent_safely failed after all retries")
                return {"error": error_msg}

    return {"error": "All retry attempts exhausted"}


from fastapi import Request

@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = None
        data_file = None

        for key, val in form.items():
            if hasattr(val, "filename") and val.filename:  # it's a file
                fname = val.filename.lower()
                if fname.endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                try:
                    if PIL_AVAILABLE:
                        image = Image.open(BytesIO(content))
                        image = image.convert("RGB")  # ensure RGB format
                        df = pd.DataFrame({"image": [image]})
                    else:
                        raise HTTPException(400, "PIL not available for image processing")
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")  
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            # Pickle for injection
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        # Build rules based on data presence
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            "Respond with the JSON object only."
        )

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process key mapping & type casting
        if keys_list and type_map:
            mapped = {}
            for idx, q in enumerate(result.keys()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        val = result[q]
                        if isinstance(val, str) and val.startswith("data:image/"):
                            # Remove data URI prefix
                            val = val.split(",", 1)[1] if "," in val else val
                        mapped[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        mapped[key] = result[q]
            result = mapped

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - Retries up to 3 times if agent returns no output.
    - If pickle_path is provided, injects that DataFrame directly.
    - If no pickle_path, falls back to scraping when needed.
    """
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}
        # return results_dict

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


    
from fastapi.responses import FileResponse, Response
import base64, os

# 1×1 transparent PNG fallback (if favicon.ico file not present)
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST /api for actual analysis."""
    health_info = {}
    if hasattr(llm, 'get_health_status'):
        health_info = llm.get_health_status()
    
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
        "llm_health": health_info
    })

@app.get("/api/health")
async def api_health():
    """Detailed API health status endpoint"""
    if hasattr(llm, 'get_health_status'):
        return JSONResponse(llm.get_health_status())
    else:
        return JSONResponse({"error": "Health status not available"})



# -----------------------------
# System Diagnostics
# -----------------------------
# ---- Add these imports near other imports at top of app.py ----
import asyncio
import httpx
import importlib.metadata
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime, timedelta
import socket
import platform
import psutil
import shutil
import tempfile
import os
import time 
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse    

# ---- Configuration for diagnostics (tweak as needed) ----
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30  # seconds per key/model simple ping test (sync tests run in threadpool)
DIAG_PARALLELISM = 6       # how many thread workers for sync checks
RUN_LONGER_CHECKS = False  # Playwright/duckdb tests run only if true (they can be slow)

# Use existing GEMINI_KEYS / MODEL_HIERARCHY from your app. If not defined, create empty lists.
try:
    _GEMINI_KEYS = GEMINI_KEYS
    _MODEL_HIERARCHY = MODEL_HIERARCHY
except NameError:
    _GEMINI_KEYS = []
    _MODEL_HIERARCHY = []

# helper: iso timestamp
def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

# helper: run sync func in threadpool and return result / exception info
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)
async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        # re-raise for caller to capture stacktrace easily
        raise

# ---- Diagnostic check functions (safely return dicts) ----
def _env_check(required=None):
    required = required or []
    out = {}
    for k in required:
        out[k] = {"present": bool(os.getenv(k)), "masked": (os.getenv(k)[:4] + "..." + os.getenv(k)[-4:]) if os.getenv(k) else None}
    # Also include simple helpful values
    out["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    return out

def _system_info():
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    # disk free for app dir and tmp
    try:
        _cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(_cwd).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
    except Exception:
        info["tmp_free_gb"] = None
    # GPU quick probe (if torch installed)
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info

def _temp_write_test():
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tmp, "write_ok": ok}

def _app_write_test():
    # try writing into current working directory
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}

def _pandas_pipeline_test():
    import pandas as _pd
    df = _pd.DataFrame({"x":[1,2,3], "y":[4,5,6]})
    df["z"] = df["x"] * df["y"]
    agg = df["z"].sum()
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(agg)}

def _installed_packages_sample():
    # return top 20 installed package names + versions
    try:
        out = []
        for dist in importlib.metadata.distributions():
            try:
                out.append(f"{dist.metadata['Name']}=={dist.version}")
            except Exception:
                try:
                    out.append(f"{dist.metadata['Name']}")
                except Exception:
                    continue
        return {"sample_packages": sorted(out)[:20]}
    except Exception as e:
        return {"error": str(e)}

def _network_probe_sync(url, timeout=30):
    # synchronous network probe for threadpool use
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- LLM key+model light test (sync) ----
# tries each key for each model with a short per-call timeout (run in threadpool)
def _test_gemini_key_model(key, model, ping_text="ping"):
    """
    Test a Gemini API key by sending a minimal request.
    Always returns a pure dict with only primitive types.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        return {"ok": False, "error": f"langchain_google_genai import error: {e}"}

    try:
        obj = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key
        )

        def extract_text(resp):
            """Normalize any type of LLM response into a clean string."""
            try:
                if resp is None:
                    return None
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    return resp.content
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                # For objects with .dict() method
                if hasattr(resp, "dict"):
                    try:
                        return str(resp.dict())
                    except Exception:
                        pass
                return str(resp)
            except Exception as e:
                return f"[unreadable response: {e}]"

        # First try invoke()
        try:
            resp = obj.invoke(ping_text)
            text = extract_text(resp)
            return {"ok": True, "model": model, "summary": text[:200] if text else None}
        except Exception as e_invoke:
            # Try __call__()
            try:
                resp = obj.__call__(ping_text)
                text = extract_text(resp)
                return {"ok": True, "model": model, "summary": text[:200] if text else None}
            except Exception as e_call:
                return {"ok": False, "error": f"invoke failed: {e_invoke}; call failed: {e_call}"}

    except Exception as e_outer:
        return {"ok": False, "error": str(e_outer)}

# ---- Async wrappers that call the sync checks in threadpool ----
async def check_network():
    coros = []
    for name, url in DIAG_NETWORK_TARGETS.items():
        coros.append(run_in_thread(_network_probe_sync, url, timeout=30))
    results = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)
    out = {}
    for (name, _), res in zip(DIAG_NETWORK_TARGETS.items(), results):
        if isinstance(res, Exception):
            out[name] = {"ok": False, "error": str(res)}
        else:
            out[name] = res
    return out

async def check_llm_keys_models():
    """Try all GEMINI_KEYS on each model (light-touch). Runs in threadpool with per-key timeout."""
    if not _GEMINI_KEYS:
        return {"warning": "no GEMINI_KEYS configured"}

    results = []
    # we will stop early if we find a working combo but still record attempts
    for model in (_MODEL_HIERARCHY or ["gemini-2.5-pro"]):
        # test keys in parallel for this model
        tasks = []
        for key in _GEMINI_KEYS:
            tasks.append(run_in_thread(_test_gemini_key_model, key, model, timeout=DIAG_LLM_KEY_TIMEOUT))
        completed = await asyncio.gather(*[asyncio.create_task(t) for t in tasks], return_exceptions=True)
        model_summary = {"model": model, "attempts": []}
        any_ok = False
        for key, res in zip(_GEMINI_KEYS, completed):
            if isinstance(res, Exception):
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, "ok": False, "error": str(res)})
            else:
                # res is dict returned by _test_gemini_key_model
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, **res})
                if res.get("ok"):
                    any_ok = True
        results.append(model_summary)
        if any_ok:
            # stop once first model has a working key (respecting MODEL_HIERARCHY)
            break
    return {"models_tested": results}

# ---- Optional slow heavy checks (DuckDB, Playwright) ----
async def check_duckdb():
    try:
        import duckdb
        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}
        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:
        return {"duckdb_error": str(e)}

async def check_playwright():
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:
        return {"playwright_error": str(e)}

# ---- Final /diagnose route (concurrent) ----
from fastapi import Query

@app.get("/summary")
async def diagnose(full: bool = Query(False, description="If true, run extended checks (duckdb/playwright)")):
    started = datetime.utcnow()
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None
    }

    # prepare tasks
    tasks = {
        "env": run_in_thread(_env_check, ["gemini_api_1","gemini_api_2","gemini_api_3","gemini_api_4","gemini_api_5","gemini_api_6","gemini_api_7","gemini_api_8","gemini_api_9","gemini_api_10","GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS"], timeout=30),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models())
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

    # run all concurrently, collect results
    results = {}
    for name, coro in tasks.items():
        try:
            res = await coro
            results[name] = {"status": "ok", "result": res}
        except TimeoutError:
            results[name] = {"status": "timeout", "error": "check timed out"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    report["checks"] = results

    # quick summary flags
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    if failed:
        report["status"] = "warning"
        report["summary"]["failed_checks"] = failed
    else:
        report["status"] = "ok"
        report["summary"]["failed_checks"] = []

    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
