# AfriGuard Multilingual Safety

Language-Aware Multilingual Adaptation for Robust Safety Classification in African Languages: A Study on AfriGuard.

Entry for the AfricaNLP 2027 AfriGuard shared task (binary safety and ten-category safety classification across ten African languages, generative models up to 5B parameters).

## Status
Stage 1: research foundation and environment setup.

## Setup

make setup # uv sync --extra dev
make test
make env

Training dependencies are optional: `uv sync --extra train` (GPU host only).

## Data
Competition data is not redistributed here. See `data/README.md`.
