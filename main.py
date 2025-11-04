from src.preprocessing import EmotionDataPreprocessor
from src.emotion_distribution import EmotionDistribution
from src.train_lora_llm import LoraFineTuner
from src.train_prompting import PromptTrainer
from src.eval_model import LoRAModelEvaluator
from utils.utils import heading
import yaml
import os


def main():
    # ============================================================
    #  LOAD CONFIGURATION
    # ============================================================
    heading("Loading configuration", char="=", color="cyan", pad=1)
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    heading("Configuration settings", char="=", color="green", pad=1)
    print(yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False))

    general_cfg = cfg["GENERAL"]
    emotion_cfg = cfg["EMOTION_ANALYSIS"]
    lora_cfg = cfg["LORA"]
    prompt_cfg = cfg["PROMPTING"]
    eval_cfg = cfg.get("EVALUATION")

    os.makedirs(general_cfg["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(general_cfg["PLOTS_DIR"], exist_ok=True)
    os.makedirs(general_cfg["MODELS_DIR"], exist_ok=True)

    # ============================================================
    #  EMOTION ANALYSIS (plots)
    # ============================================================
    if emotion_cfg.get("ENABLE_PLOTTING", False):
        heading("Analyzing emotion data", char="=", color="green", pad=1)
        for lang in emotion_cfg.get("LANGUAGES", []):
            input_path = os.path.join(general_cfg["DATA_DIR"], f"{lang}-projections.tsv")
            output_path = os.path.join(
                general_cfg["PLOTS_DIR"], f"emotion_distribution_{lang}.png"
            )
            EmotionDistribution(input_path, output_path).plot_distribution(save=True)
        print(f"Saved emotion distribution plots to '{general_cfg['PLOTS_DIR']}/'")

    # ============================================================
    #  PREPROCESSING
    # ============================================================
    heading("Emotion Data Preprocessing", char="=", color="cyan", pad=1)
    preprocessor = EmotionDataPreprocessor(
        data_dir=general_cfg["DATA_DIR"],
        output_dir=general_cfg["OUTPUT_DIR"],
        test_size=general_cfg["TEST_SIZE"],
        seed=general_cfg["SEED"],
    )
    preprocessor.run()
    heading("Preprocessing Complete", char="=", color="green", pad=1)

    # ============================================================
    #  LORA FINE-TUNING
    # ============================================================
    if lora_cfg.get("LORA_FINETUNE", False):
        heading("Training model(s) with LoRA", char="=", color="cyan", pad=1)
        languages = emotion_cfg.get("LANGUAGES", [])
        data_paths = {}
        for lang in languages:
            data_paths[lang] = {
                "train": os.path.join(general_cfg["OUTPUT_DIR"], lang, "train.jsonl"),
                "validation": os.path.join(general_cfg["OUTPUT_DIR"], lang, "validation.jsonl"),
                "test": os.path.join(general_cfg["OUTPUT_DIR"], lang, "test.jsonl"),
            }

        print(f"Detected language datasets: {list(data_paths.keys())}")

        trainer = LoraFineTuner(
            data_paths=data_paths,
            models=lora_cfg["LORA_MODELS"],
            output_dir=general_cfg["MODELS_DIR"],
            num_epochs=lora_cfg["NUM_EPOCHS"],
            batch_size=lora_cfg["BATCH_SIZE"],
            grad_accum_steps=lora_cfg["GRAD_ACCUM_STEPS"],
            learning_rate=lora_cfg["LEARNING_RATE"],
            device_ids=lora_cfg["DEVICE_IDS"],
        )
        trainer.run_all()
        heading("LoRA Fine-tuning Complete", char="=", color="green", pad=1)
    else:
        heading("LoRA fine-tuning disabled in config.yaml", char="=", color="red", pad=1)
    
            
    #  ============================================================
    # MODEL EVALUATION
    #  ============================================================
    if eval_cfg.get("ENABLED", False):
        evaluator = LoRAModelEvaluator(
            base_model=eval_cfg["BASE_MODEL"],
            adapter_path=eval_cfg["ADAPTER_PATH"],
            batch_size=eval_cfg["BATCH_SIZE"],
            max_new_tokens=eval_cfg["MAX_NEW_TOKENS"],
        )
        results = evaluator.evaluate(
            data_root=general_cfg["OUTPUT_DIR"],
            langs=emotion_cfg.get("LANGUAGES", [])
        )
        
        print("Evaluation Results:")
        print(results)
        

    # ============================================================
    #  PROMPTING EVALUATION
    # ============================================================
    if prompt_cfg.get("ENABLE_PROMPTING", False):
        heading("Evaluating Prompting Strategies", char="=", color="cyan", pad=1)

        # build paths from config languages
        languages = emotion_cfg.get("LANGUAGES", [])
        data_dirs = {
            lang: os.path.join(general_cfg["OUTPUT_DIR"], lang, "test.jsonl") for lang in languages
        }

        prompt_trainer = PromptTrainer(
            model_name=prompt_cfg["MODEL_NAME"],
            data_dirs=data_dirs,
            max_new_tokens=prompt_cfg["MAX_NEW_TOKENS"],
            device=prompt_cfg.get("DEVICE", None),
            output_dir=general_cfg["MODELS_DIR"],
            instruction_text=prompt_cfg["INSTRUCTION_TEXT"],
            few_shot_intro=prompt_cfg.get(
                "FEW_SHOT_INTRO",
                "You are an emotion classifier. Here are examples:",
            ),
            few_shot_examples=prompt_cfg.get("FEW_SHOT_EXAMPLES", None),
        )
        prompt_trainer.run()

        heading("Prompting Evaluation Complete", char="=", color="green", pad=1)

if __name__ == "__main__":
    main()
