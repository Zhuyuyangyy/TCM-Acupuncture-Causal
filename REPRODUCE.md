# REPRODUCE.md - TCM-Acupuncture-Causal

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Not required (CPU sufficient for API server)

## Install

```bash
cd TCM-Acupuncture-Causal
pip install -e .
```

Or install dependencies directly:
```bash
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

## Smoke Test

```bash
python -c "from backend.main import app; print('Import OK')"
```

```bash
pytest backend/tests/ -v
```

## Run Server

```bash
uvicorn backend.main:app --port 8020 --reload
```

## API Documentation

Access Swagger UI at: http://localhost:8020/docs

## Project Description

针灸真实世界因果推断与个体化穴位组合推荐

## Known Issues

- No external real clinical data included; uses synthetic/demo data
- torch is a heavy dependency; consider CPU-only install for API-only usage
- No hardcoded absolute paths detected in core code
