# Development

Use Python 3.10 or later with compatible research dependencies. Run experiments directly with `python main.py` from the repository root; the project does not need to be installed. Run `python -m pytest -q` after a change.

Keep `configs/examples/*.json` as illustrative templates. Create your own experiment configuration by copying and adapting a template with a recognized task, explicit parameters, environment and unique run directory. Inspect the generated command with `--dry-run`. Change `run_dir` and all output directories when comparing seeds; never overwrite an earlier experiment's artifacts.

The launcher delegates to the original scripts so that existing architecture/checkpoint formats remain usable. Model timing, device deployment and classification evaluation are separate operations. A new measurement backend should record candidate IDs, units, batch size, synchronization, warmup and repeat counts. A new FPGA/GPU backend must provide its actual toolchain; an analytical estimate is not a deployed measurement.

Use small synthetic arrays only for mechanical tests and label them as test fixtures. Do not populate manuscript-result tables with test-generated metrics. Keep reported values, provenance and confidence statements aligned with the source experiment.

When submitting a change, include the affected configuration, relevant validation and limitations. Keep datasets, local caches, checkpoint weights and run logs out of ordinary Git commits. The repository-wide license remains to be confirmed by the authors.
