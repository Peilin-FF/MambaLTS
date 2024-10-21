if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
model_name=Mamba2_Norm
for norm_type in IGWItN BGN GN BN DBN ItN IN ItNIns None Scale
do
  for pred_len in 96 192 336 720
  do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path /root/Time-Series-Library/dataset/electricity \
    --data_path electricity.csv \
    --model_id ECL_$pred_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $pred_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 321 \
    --expand 1 \
    --d_ff 16 \
    --d_conv 4 \
    --c_out 321 \
    --d_model 128 \
    --des 'Exp' \
    --norm_type $norm_type \
    --T 5 \
    --num_channels 8 \
    --num_groups 4 \
    --runname exchange_rate'_'$pred_len'_'$norm_type \
    --itr 1 >> logs/ILI_Mamba2_Norm.log

  done
done