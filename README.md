# ZSTCI
 Incremental Embedding Learning via Zero-Shot Translation,AAAI Conference on Artificial Intelligence (__AAAI__), 2021.
 
 Requirments
 
All training and test are done in Pytorch framework.

Pytorch vesion: 0.4.1

Python version: 3.6

Datasets

We evaluate our methods in CUB-200-2011 and CIFAR100. (Note: CUB-200-2011 do not split the train set and test set in the original folder, the splited datasets can be download from this [link]（https://drive.google.com/drive/folders/1sjJTCbVriYSbntQfGMQUJH7y2D_UogT2） according to the original provided train/test text file.)

Train and test

run CUB:

bash  run_demo_cub.sh

run Cifar100:

bash  run_demo_cifar.sh
 
 




Acknowledgements

Our code structure is inspired by [SDC](https://github.com/yulu0724/SDC-IL).
