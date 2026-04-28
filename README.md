# NoMoreSwearing Infra MVP

Minimal standalone inference package for the NoMoreSwearing Qwen DoRA classifier.

## API

```bash
python main.py
```

or:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"text to classify\"]}"
```

Runtime parameters are read from `config.json`, not command-line flags. The local
adapter and classifier head weights live in `qwen-dora-db/`. The base model
defaults to `Qwen/Qwen3.5-0.8B-Base` from `model_metadata.json`, so the runtime
needs that model available from Hugging Face or local cache.
