# NetvirE Chatbot (SmartAssist)

This repo contains:
- **FastAPI backend**: `app/main.py` (runs on port `8088`)
- **Streamlit UI**: `app/smart_iot_app.py`

## Run locally (Windows PowerShell)

Create/activate venv, install deps:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run FastAPI:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8088
```

Run Streamlit:

```powershell
streamlit run .\app\smart_iot_app.py
```

## Docker

Build:

```powershell
docker build -t netvire-chatbot:latest -f Dockerfile .
```

Run:

```powershell
docker run --rm -p 8088:8088 netvire-chatbot:latest
```


