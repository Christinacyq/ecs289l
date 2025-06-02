# ecs289l
python main_train.py \
    --image_dir data/images \
    --ann_path data/annotation.json \
    --output_dir results \
    --batch_size 16 \
    --epochs 1 \
--learning_rate 1e-4

check the main_train for hyperparameters tuning
download pycocoevalcap into "pycocoevalcap" folder: Download the pycocoevalcap package from https://drive.google.com/drive/folders/1WFxcn2G2bUG-bp7pMEXKxrFWyk1rZmbh?usp=drive_link and place them in this directory.
images into data/images

primaryly use visual_extractor.py for encoder