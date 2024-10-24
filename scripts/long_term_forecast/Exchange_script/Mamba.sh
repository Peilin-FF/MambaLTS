model_name=Mamba
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /root/Time-Series-Library/dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id Exchange_$pred_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 8 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 8 \
  --d_model 128 \
  --des 'Exp' \
  --runname exchange'_'rate'_'$pred_len'_'shuffle=True \
  --itr 1 >> logs/exchange_rate_Mamba.log

done