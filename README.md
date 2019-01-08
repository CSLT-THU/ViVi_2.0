# ViVi_2.0

## Introduction

ViVi_2.0 is the code of the Chinese poem generation system, which can generate different styles of Chinese poems based on memory mechanism.

## User Manual

### Installation

#### System Requirements

* Linux or MacOS
* Python 2.7

We recommand to use GPUs:

* NVIDIA GPUs 
* cuda 7.5

#### Installing Prerequisites

##### CUDA 7.5 environment
Assume CUDA 7.5 has been installed in "/usr/local/cuda-7.5/", then environment variables need to be set:

```
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH 
```
##### Tensorflow 0.10
To have tensorflow 0.10 installed, serval methods can be applied. Here, we only introduce the installation through virtualenv. And we install the tensorflow-gpu, if you choose to use CPU, please install tensorflow of cpu.

```
pip install virtualenv --user
virtualenv --system-site-packages tf0.10  
source tf0.10/bin/activate
download tensorflow_0.10 from https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
pip install tensorflow_0.10
```
##### Test installation
Get into python console, and import tensorflow. If no error is encountered, the installation is successful.

```
Python 2.7.5 (default, Nov  6 2016, 00:28:07) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-11)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow 
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
>>> 
```

### Quick Start
If you are from cslt, you can read run_in_cslt.txt and run the program on servers of cslt.

If you are not from cslt, make sure you have finished the installations above, then follow the commands below.

Prepare:
```
git clone git@gitlab.com:cslt/vivi_2.0.git
cd vivi_2.0
```

Train a model:
```
python train/train.py --decode False --batch_size 80
```

The models which are generated when the code runs through a round will be Saved in the dir 'tmp'.


Run a selected model:
```
python predict/predict.py model_name style memory_weight round_num poem_type
```
`model_name` : the model which is selected in the dir 'train/tmp/' is put in 'predict/model/'.

`style` : biansai/tianyuan/yanqing/general. Illegal input will be considered as 'general'.

`memory_weight` : suggest 0-30.

`round_num` : suggest 1.

`poem_type` : poem5/poem7/ymr/dlh/jzmlh/djc/zgt/psm/yja. Illegal input will be considered as 'poem7'.

[五言/七言/虞美人/蝶恋花/减字木兰花/点绛唇/鹧鸪天/菩萨蛮/渔家傲]


Change test keyword input: modify the file 'resource/predict_resource/test_poem_58k.txt'.
Run examples: the file 'predict/execute.sh' includes all command. you can modify the command quickly and run it.

Get memory:
```
python predict/memoryModule_decoder.py model_name style
```
Add new style/change file for memory: modify in function 'get_memory_options()', file 'data_utils.py'. Put the poetry file under the directory 'resource/memory_resource/text/'.


### Module Instruction

Basically the poem generation system consists of two modules: `train` and `predict`.
- `train` : Train the poem generation model using seq2seq and attention mechanism. models generated are saved in dir 'train/tmp'
- `predict` : Use generated model along with memory to decode and generate samples. Models used have to be transferred to dir 'predict/model'
- `resource` : Store different sources from word2vec data to augmented memory.

## License

Open source licensing is under the Apache License 2.0, which allows free use for research purposes. 
For commercial licensing, please email byryuer@gmail.com.

## Development Team

Project leaders: Dong Wang

Project members: Jiyuan Zhang, Shiyue Zhang, Zheling Zhang, Yibo Liu, Xiuqi Jiang

## Contact

If you have questions, suggestions and bug reports, please email [xiuqi6666@gmail.com].
