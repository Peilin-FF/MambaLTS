if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
model_name=Mamba2_Norm
for norm_type in IGWItN BGN GN BN DBN ItN IN ItNIns None Scale
do
  for pred_len in 96 192 336 720
  do
    for headdim in 16 8 4
    do
      for d_ff in 16 8 4
      do
        for n_layers in 1 2 3 4
        do
        python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path /root/Time-Series-Library/dataset/ETT-small \
          --data_path ETTm1.csv \
          --model_id ETTm1_$pred_len'_'$pred_len \
          --model $model_name \
          --data ETTm1 \
          --features M \
          --seq_len $pred_len \
          --label_len 48 \
          --pred_len $pred_len \
          --e_layers 2 \
          --enc_in 7 \
          --expand 1 \
          --d_ff $d_ff \
          --d_conv 4 \
          --c_out 7 \
          --d_model 128 \
          --des 'Exp' \
          --norm_type $norm_type \
          --T 5 \
          --num_channels 8 \
          --num_groups 4 \
          --n_layers $n_layers \
          --headdim $headdim \
          --runname ETTm1'_'$pred_len'_'$norm_type'_'$headdim'_'$d_ff'_'$n_layers \
          --itr 1 >> logs/ETTm1_Mamba2_Norm.log
        done
      done
    done
  done
done
