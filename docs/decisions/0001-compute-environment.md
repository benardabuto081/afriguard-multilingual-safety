# 0001 - Compute environment

- Date: 2026-11-24
- Context: local laptop (Lenovo T470s) has no CUDA GPU.
- Decision: develop, audit data and test on CPU locally; run model training and GPU inference on an external GPU host.
- Open: choice of GPU host and its VRAM, decided at Stage 3 before any fine-tuning.
