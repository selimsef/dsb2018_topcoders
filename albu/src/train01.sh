CUDA_VISIBLE_DEVICES=0 python bowl_train.py $1 --training --fold=0
CUDA_VISIBLE_DEVICES=0 python bowl_train.py $1 --training --fold=1
