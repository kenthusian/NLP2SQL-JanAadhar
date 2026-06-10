"""
main.py — Entrypoint for the Jan-Aadhaar NL2SQL backend.

Usage:
    python main.py              # Start FastAPI on port 8000
    python main.py --port 9000  # Custom port
    python main.py --skip-preflight  # Skip startup checks

The Streamlit UI is started separately:
    python -m streamlit run ui/streamlit_app.py --server.port 8501
"""
import argparse
import sys

import uvicorn

from config import FASTAPI_PORT, OLLAMA_URL, OLLAMA_MODEL
from logger import log_event, get_logger

log = get_logger("nl2sql.main")


def preflight_check() -> bool:
    """
    Verify all external dependencies are available before starting.
    Returns False (and prints helpful messages) if any check fails.
    """
    ok = True

    # Ollama check
    try:
        import httpx
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"[WARN] Model '{OLLAMA_MODEL}' not found in Ollama.")
            print(f"       Pull it with: ollama pull {OLLAMA_MODEL}")
            ok = False
        else:
            print(f"[OK] Ollama: {OLLAMA_MODEL} is available")
    except Exception:
        print(f"[FAIL] Ollama is not reachable at {OLLAMA_URL}")
        print("       Start Ollama: ollama serve")
        print("       Pull model:   ollama pull qwen2.5-coder:3b")
        ok = False

    # Data directory check
    from config import DATA_DIR
    parquet_files = list(DATA_DIR.glob("**/*.parquet"))
    if not parquet_files:
        print(f"[WARN] No Parquet files found in {DATA_DIR}")
        print("       Ingest CSV:  python -m db.ingest csv --source <path>")
        print("       Mock data:   python -m db.ingest mock --rows 1000000")
        ok = False
    else:
        print(f"[OK] Data: {len(parquet_files)} Parquet file(s) in {DATA_DIR}")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Jan-Aadhaar NL2SQL Backend")
    parser.add_argument("--port", type=int, default=FASTAPI_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip preflight checks")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("  Jan-Aadhaar NL2SQL Pipeline  v2.0")
    print("=" * 60)

    if not args.skip_preflight:
        preflight_check()

    log_event("server_start", host=args.host, port=args.port)
    print(f"\n[START] API server -> http://{args.host}:{args.port}")
    print(f"[DOCS]  Swagger UI -> http://{args.host}:{args.port}/docs")
    print(f"\n[UI]    Run Streamlit in a new terminal:")
    print(f"        python -m streamlit run ui/streamlit_app.py --server.port 8501")
    print("")

    uvicorn.run(
        "ui.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
