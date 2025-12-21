dataset='imagenet100'
version='v1.1'

python 1-search_first_stage.py \
--mode "Search" \
--arch Predict_harsbnn \
--model_path './net_predict_output/imagenet100/CharRNN_OUR_best.pth' \
--max-epochs 20 \
--population-num 512 \
--m-prob 0.4 \
--crossover-num 128 \
--mutation-num 128 \
--seed 2024 \
--logdir ./search_output/${dataset}/${version}/ \
--layer_len_start 12 \
--layer_len_end 32 \
--baseline_arch 'mobilenet' \
--dataset 'imagenet100' \
--acc_width 224 \
--search_width 224 \
--ops-min 80_000_000 \
--ops-max 90_000_000 \
--parm-min 3_000_000 \
--parm-max 4_500_000 \
--latency-min 2.2 \
--latency-max 2.5 \
--interval 500 \
--save_epochs "5,10,15,20" \
--gpu 5 \
--first_float 1 \
--time_mode "None" \
