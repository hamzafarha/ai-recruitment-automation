# Setup

The setup guide is the [README](../README.md): install (sections 1–3), Docker services (section 4), datasets and
evaluation set (sections 5–6), checks (section 7) and **[Try everything](../README.md#8-try-everything)**
(section 8: the Kaggle AI server, the API, the interface, demo data and n8n).

Short version, once installed:

```powershell
docker compose up -d postgres qdrant n8n
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000     # terminal 1
cd web; npm install; npm run build                          # once: the API serves the interface at http://localhost:8000
```

The AI models run in the Kaggle notebook `notebooks/kaggle/model_server.ipynb` (README section 8, step 1); paste
the four lines it prints into `.env` and restart the API.
