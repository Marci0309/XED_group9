# Project structure
```bash
├── data
├── !OriginalData
├── habrok_outputs
├── jobs
│   ├── python_eval_model.sh
│   ├── python_test_model.sh
│   ├── python_train_lora.sh
│   └── python_train_prompting.sh
├── logs
│   ├── lora_mistral_24933056.out
│   ├── prompting_24932727.out
│   ├── prompting_24964901.err
│   └── prompting_24964901.out
├── models
├── notebooks
│   ├── huggingface_guide.ipynb
│   └── PEFT_Tutorial.ipynb
├── README.md
├── requirements.txt
├── results
├── main.py
└── src
    ├── eval_model.py
    ├── preprocessing.py
    ├── test_model.py
    ├── train_lora_llm.py
    └── train_prompting.py
```

# The project

This project aims to fine tune existing LLMs on the dataset Helsinki-NLP/XED. 
The goal is to use Lora finetuning and prompt engineering to familiarize an existing LLM with the data such that itcan accurately predict the labels in multiple languages.
For this task we use models of the Mistral family for funetuning and loading these models from HuggingFace

# How to run
1. PLace the .tsv files accuired from https://github.com/Helsinki-NLP/XED into the !OriginalData folder

2. Config the .yaml configuration to meet the your required training and evaluation methods.

3. Run python main.py

