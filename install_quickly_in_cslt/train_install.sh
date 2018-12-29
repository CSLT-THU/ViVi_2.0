#!/usr/bin/env bash
pip install virtualenv --user

export PATH="/nfs/disk/work/users/$USER/.local/bin:$PATH"

#virtualenv --no-site-packages ../tf0.10  

virtualenv -p /usr/bin/python2.7 --no-site-packages ../tf0.10

source ../tf0.10/bin/activate

pip install tensorflow-0.10.0-cp27-none-linux_x86_64.whl

source /work4/zhangsy/env_enable_7.5

cd ..

python train/train.py --decode False --batch_size 80
