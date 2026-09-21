import subprocess
import time

backend_folder = "/content/drive/MyDrive/SIH_CyclonetAI_Project/backend"

server = subprocess.Popen(
    [
        "uvicorn",
        "main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ],
    cwd=backend_folder
)

time.sleep(5)

print("✅ Cyclonet-AI FastAPI server started!")
print("📡 Running on port 8000")
