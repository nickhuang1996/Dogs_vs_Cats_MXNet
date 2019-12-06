# Dogs_vs_Cats_MXNet
# Introduction
- This repository is for kaggle `Dogs vs. Cats` match, but you can utilize this code to learn how to use `mxnet`. 
- For network, I has estabilished the structure containing the introduction of pre-trained models like `VGG` and `ResNet`.
- For sampler, there are 2 types which are `Sequential` and `Random`.
- For lr_scheduler, I has coded 4 types to adjust the learning rate for the optimizer.
- For optimizer, `Adam` and `SGD` are only illustrated in my repository.

## Environment
- Python 3.6
- mxnet-cu90
- tqdm 4.28.1
- tensorboardX 1.5
    
## Dataset Structure
### Original
```
${project_dir}/datasets
    dogs-vs-cats
        train.zip
        test1.zip
```
### Extract train and test datasets
After downloading the datasets from Kaggle website, you need to extract these two zips.(Actually, I just extract train.zip)
```
${project_dir}/datasets
    dogs-vs-cats
        train.zip
        test1.zip
        train           # Extracted from train.zip
        test1           # Extracted from test1.zip
```
### Final step
- In fact, the filenames of train and test datasets is in different naming conventions.
- For train one, the filename name is in cat.x.jpg or dog.x.jpg. However, x.jpg is used in test dataset. 
- To do the classification task easily, I just use the train dataset. So if you want to build a new test dataset. You need to run `redistribution_image.py` to split the train dataset into a new one and an extra test dataset.
```
${project_dir}/datasets
    dogs-vs-cats
        train.zip
        test1.zip
        test            # Separated from 'train' directory by run `redistribution_image.py`.
        train           # Extracted from train.zip
        test1           # Extracted from test1.zip
```
## Experimental Directory Structure
- Before training, you need to modify the directories in `parser_setting.py`
- Run `demo.py` to start the training process. The follow directories will be created automatically.
```
${weight_results_dir}
    Dogs_vs_Cats_MXNET
        checkpoints
            RES
                0.params
                1.params
                ...
                9.params
                best.params
            VGG
                0.params
                1.params
                ...
                9.params
                best.params
        log
            RES
                test
                train
            VGG
                test
                train
        pretrained_models
            resnet50_v2-ecdde353.params
            vgg16_bn-7f01cf05.params
            ...
        tensorboard
            IDLoss
            ...
```
## TensorboardX
- You can walk into `tenserboard` directory to monitor the loss. Run
`tensorboard --logdir .` then open the browser.
## Performances
| Network | Accuracy(%)|
|---|---|
| VGG16_bn | 97.92 |
| ResNet50_v2 | 97.84 |
- I has just trained the models for 10 epochs by 'Adam'.
