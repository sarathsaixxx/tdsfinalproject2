import subprocess
import sys
import traceback
from typing import List

async def run_python_code(code: str, libraries: List[str], folder: str = "uploads") -> dict:
    import os
    import io

    def execute_code():
        # Ensure working directory is the request folder
        #os.makedirs(folder, exist_ok=True)
        #old_cwd = os.getcwd()
        #os.chdir(folder)
        #try:
        #    exec_globals = {}
        #    exec(code, exec_globals)
        #finally:
        #    os.chdir(old_cwd)

        exec_globals = {}
        exec(code, exec_globals)

    # Step 1: Install all required libraries first
    for lib in libraries:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except Exception as install_error:
            return {"code": 0, "output":f"❌ Failed to install library '{lib}':\n{install_error}"}

    # Step 2: Execute the code after installation
    try:
        execute_code()
        return {"code": 1, "output": "✅ Code executed successfully after installing libraries."}

    except Exception as e:
        return {"code": 0, "output": f"❌ Error during code execution:\n{traceback.format_exc()}" }