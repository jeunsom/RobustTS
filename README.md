# RobustTS
Robust Time Series Recovery and Classification Using Test-Time Noise Simulator Networks

## Overview
This is re-implementation of the RobustTS for ICASSP 2023

## Requirements
* pytorch>=1.6.0
* python>=3.6.0

## Time Series Data
We use HDM05 as an example with AE architecture for RobustTS. To reproduce the results described on the paper, please modify the hyperparameters in HDM_corruption_test/run_main_ts_hdm.py. The users can also change the data to other dataset at their interest.

## Sample
* Pre-trained decoder model can be downloaded: https://www.dropbox.com/sh/3p28i85ypj118is/AADLKheQbrHXUTBIbuScWI-Sa?dl=0
* RobustTS on HDM05 in HDM_shift_test:
python3 run_main_ts_hdm_shift_scale_test.py --dim_z 200 --trial hmd_loss1_200_action --batch_size 32 --encoder-checkpoint encoder_loss1_200hmd_ae.pth.tar --decoder-checkpoint decoder_loss1_200hmd_ae.pth.tar --save False --results_path hmd_fold1_shift1 --num_epochs 20 --add_noise 0 --shift 1 --test_id 1

* RobustTS on HDM05 in HDM_corruption_test:
python3 run_main_ts_hdm.py --dim_z 200 --trial hmd_loss1_200_action --batch_size 32 --encoder-checkpoint encoder_loss1_200hmd_ae.pth.tar --decoder-checkpoint decoder_loss1_200hmd_ae.pth.tar --save False --results_path hmd_loss1_200_ae_action_fold1_rnd1 --num_epochs 20 --add_noise 7 --shift 1 --save_corrupted_train 1 --test_id 1
* RobustTS++ (2nd Step, Recovered Train Data):
python3 run_main_ts_hdm_2ndstep.py --dim_z 200 --trial hmd_loss1_200_action --batch_size 32 --encoder-checkpoint encoder_loss1_200hmd_ae.pth.tar --decoder-checkpoint decoder_loss1_200hmd_ae.pth.tar --save False --results_path hmd_loss1_200_ae_action_fold1_2ndstep --num_epochs 20
* RobustTS++ (3rd Step, Further Refinement):
python3 run_main_ts_hdm_3rdstep.py --dim_z 200 --trial hmd_loss1_200_action_3rdstep --batch_size 32 --save False --results_path hmd_loss1_200_ae_action_fold1_3rdstep --num_epochs 200
