if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
model_name=Mamba2_RMS

for pred_len in 96 192 336 720
do
  for headdim in 16 8 4
  do
    for n_layers in 1 2 3 4 5 6 7 8
    do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path /root/Time-Series-Library/dataset/ETT-small \
      --data_path ETTm2.csv \
      --model_id ETTm2_$pred_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $pred_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 7 \
      --expand 1 \
      --norm_type 'None' \
      --T 5 \
      --num_channels 8 \
      --num_groups 4 \
      --n_layers $n_layers \
      --headdim $headdim \
      --d_ff 16 \
      --d_conv 4 \
      --c_out 7 \
      --d_model 128 \
      --des 'Exp' \
      --runname ETTm2'_'$pred_len'_'$headdim'_'$n_layers \
      --itr 1 >> logs/ETTm2_Mamba2_RMS.log
    done
  done
done
