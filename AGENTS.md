# Instructions for Codex

- This project fine-tunes the Google Gemma 3 1B instruction-tuned model using LoRA or QLoRA on Kaggle GPUs.
- Respect the original model ID `google/gemma-3-1b-it` and never hard-code tokens or other credentials.
- Before changing training logic, run `python3 -m unittest discover -s tests -v` and `python3 train.py --check`.
- Mac development must not require GPU or downloaded model weights; keep all ML imports inside `run_training`.
- Keep `train.py` as the root CLI entry point; its implementation is `src/fama_style/training/train.py`.
- Regenerate `kaggle/train.ipynb` from the training implementation using `python3 scripts/build_notebooks.py` before Kaggle submissions. The builder must keep notebooks self-contained for manual UI upload.
- Treat 3 example rows and 5 training steps as a pipeline smoke test, not an actual fine-tuning experiment.
- Do not invoke Kaggle GPU training without the user's explicit approval.
