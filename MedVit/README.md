

Add a pycocoevalcap accordingly
cd pycocoevalcap
Download the pycocoevalcap package from [here](https://drive.google.com/drive/folders/1WFxcn2G2bUG-bp7pMEXKxrFWyk1rZmbh?usp=drive_link) and place them in the `pycocoevalcap` directory.
```
Train with: 
python main_train_XRGen.py --batch_size 16 --save_dir results/XRGen

python main_train_XRGen.py --visual_extractor vit_base --d_vf 768 --batch_size 16 --save_dir results/XRGen

Add dataset accordingly:
* Put the image data under `data/images/` should be like:
```
      ├── CXR2384_IM-0942/
          ├── 0.png
          ├── 1.png
      ├── CXR2926_IM-1328/
          ├── 0.png
          ├── 1.png
```





<!-- ## Download our X-RGen weights
You can download the models we trained for our dataset from [here](https://drive.google.com/file/d/1mkT3PcrE_s9vkjqg_Vn5rdmh-z2bhfa6/view?usp=drive_link).

## Data Preparation
* We will not disclose our private data in order to protect privacy. To utilize our code, please format your data according to the specifications outlined below. <br/>

* Put the report data under `data/annotation.json` should be like the example in `data/annotation_example.json`

## Training
* Please download the pre-trained weights for MedCLIP from [here](https://storage.googleapis.com/pytrial/medclip-pretrained.zip) and place them in the `models` directory.
* Run
```
python main_train_XRGen.py --batch_size 192 --save_dir results/XRGen
```
to train a model on your data.

## Inference
* Run
```
python main_test_XRGen.py --batch_size 192 --save_dir results/XRGen --load results/XRGen/model_best.pth
```
for inference.

 -->
