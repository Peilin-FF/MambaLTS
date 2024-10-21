#!/bin/bash

./scripts/long_term_forecast/ECL_script/Mamba2_BN.sh &&
./scripts/long_term_forecast/Traffic_script/Mamba2_BN.sh &&
./scripts/long_term_forecast/Exchange_script/Mamba2_BN.sh &&
./scripts/long_term_forecast/Weather_script/Mamba2_BN.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_BN_ETTh1.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_BN_ETTh2.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_BN_ETTm1.sh &&
./scripts/long_term_forecast/ETT_script/Mamba2_BN_ETTm2.sh
