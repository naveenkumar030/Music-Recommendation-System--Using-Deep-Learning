import uvicorn

if __name__ == "__main__":
    print("Starting SoundSpace DL server on http://127.0.0.1:8000 ...", flush=True)
    uvicorn.run("src.serving.api:app", host="127.0.0.1", port=8000, log_level="info")
