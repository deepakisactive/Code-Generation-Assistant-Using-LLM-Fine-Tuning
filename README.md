# Code-Generation-Assistant-Using-LLM-Fine-Tuning
A Generative AI project that fine-tunes a TinyLlama model using QLoRA for code generation tasks. Includes dataset preprocessing, model training, evaluation (BLEU, ROUGE, Accuracy), comparison with base model, and a FastAPI-based UI for real-time code generation.
# Code Generation Assistant using LLM Fine-Tuning

## Overview
This project focuses on fine-tuning a Large Language Model (TinyLlama) for code generation tasks using a custom dataset. The model is trained using QLoRA (PEFT) and evaluated using BLEU, ROUGE, and Accuracy metrics.

## Features
- Custom dataset (1100+ samples)
- Fine-tuning using QLoRA
- Evaluation using BLEU, ROUGE, Accuracy
- Base vs Fine-tuned model comparison
- FastAPI backend for inference
- Simple frontend UI for real-time code generation

## Tech Stack
- Python
- HuggingFace Transformers
- PEFT (QLoRA)
- FastAPI
- HTML, CSS, JavaScript
- Google Colab
- VS Code


## How to Run
1. Start backend:
   uvicorn app:app --reload
2. Open `index.html` in browser

## Results
- BLEU Score: ~0.15
- ROUGE Score: ~0.42

## Author
Deepak 
