# Example configurations

These JSON files demonstrate how to configure predictor fitting, initialization, search, training, evaluation, and plotting. They are usage examples, not the official configurations for reproducing particular paper models.

Copy the relevant example and adapt the data/checkpoint paths, architecture, hyperparameters, hardware settings, output paths, and run directory for your experiment. Keep separate outputs for independent runs.

Preview the resolved command before running:

```bash
python main.py run --config configs/examples/init-imagenet100.json --dry-run
```

The initialization example preserves boundary sampling for predictor fitting: eight initial candidates plus 50 maximal candidates, with seed 1234. These values illustrate the workflow and can be changed for your own labeling plan.
