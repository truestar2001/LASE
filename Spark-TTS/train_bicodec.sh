export PYTHONPATH=$(pwd)
# python cli/train_bicodec.py
# CUDA_VISIBLE_DEVICES=0 python -c "import torch; print('count=', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
CUDA_VISIBLE_DEVICES=5 torchrun cli/train_bicodec.py