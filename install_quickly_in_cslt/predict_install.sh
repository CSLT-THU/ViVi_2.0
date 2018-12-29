#!/usr/bin/env bash
cd ..

python predict/memoryModule_decoder.py model_name style
python predict/predict.py model_name style memory_weight round_num poem_type

#examples as below

#python memoryModule_decoder.py 1th_model.ckpt+cost=181.362346255 yanqing
#python predict.py 1th_model.ckpt+cost=181.362346255 yanqing 16.0 1 poem7
