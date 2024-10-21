#!/bin/bash

./scripts/long_term_forecast/ECL_script/Mamba2_RMS.sh &&
./scripts/long_term_forecast/Traffic_script/Mamba2_RMS.sh &&
./scripts/long_term_forecast/Exchange_script/Mamba2_RMS.sh &&
./scripts/long_term_forecast/Weather_script/Mamba2_RMS.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_RMS_ETTh1.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_RMS_ETTh2.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_RMS_ETTm1.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_RMS_ETTm2.sh
