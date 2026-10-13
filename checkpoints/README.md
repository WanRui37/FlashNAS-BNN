# Checkpoints

No pretrained model files or verified public download links are distributed here. Do not substitute random weights for pretrained evaluation.

| Artifact | Default location | Format and producer |
| --- | --- | --- |
| CIF-Net predictor | `net_predict_output/imagenet100/CharRNN_OUR_best.pth` | PyTorch model `state_dict`, predictor training |
| Stage-I candidates | `search_output/imagenet100/example/stage1/info.pth` | Dictionary with `final_candidates`, accuracy/OP/parameter arrays and per-epoch snapshots |
| Timed candidates | `search_output/imagenet100/example/stage2-pytorch/info_second.pth` | Same candidate rows with `final_candidates_latency` |
| CPU bandwidth predictor | `net_bw_output/model_best.pth` | `Predict_BW` state dict, external measured-bandwidth training artifact |
| Trained network | `train_imagenet/outputs/stage2/example/checkpoint/model_best.pth.tar` | Dictionary including `state_dict` and training epoch |

Supply a matching architecture, feature preprocessing, class mapping and training budget with every checkpoint. The stage-2 evaluation example loads the trained checkpoint using the existing resume path. Preserve checkpoint SHA256 values in your experiment release. Legacy PyTorch checkpoint loading is for files you trust; compatibility with PyTorch versions that change the `torch.load` default must be validated explicitly.

Predictor test mode uses the same `out_path/<dataset>/` location as predictor training. For a separate validation run, set `parameters.mode=test` and a new `run_dir`, preserving the checkpoint output path.
