# Project structure
├── !OriginalData
│   ├── en-projections.tsv
│   ├── hu-projections.tsv
│   ├── nl-projections.tsv
│   └── ro-projections.tsv
├── data
│   ├── en
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   ├── hu
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   ├── nl
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   └── ro
│       ├── test.jsonl
│       ├── train.jsonl
│       └── validation.jsonl
├── habrok_outputs
│   ├── # all habrok output logs
├── jobs
│   ├── # all habrok job submission scripts
├── lora_mistral_finetune
├── models
│   ├── # stores all saved models
├── notebooks
│   ├── # course tutorial notebooks
├── pipeline
│   ├── eval_model.py
│   ├── preprocessing.py
│   ├── test_model.py
│   └── train_lora_llm.py
├── README.md
├── requirements.txt
