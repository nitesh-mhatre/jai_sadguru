#!/bin/bash
echo "Setting up Jai Sadguru — NIFTY F&O Expert"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Done! Run with:"
echo "  source venv/bin/activate"
echo "  python main.py                    # NVIDIA NIM (default)"
echo "  python main.py --model qwen2.5:7b # Ollama local"
