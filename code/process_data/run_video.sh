#!/bin/bash

source /opt/anaconda3/bin/activate
DATA_DIR=$1
NAME=$2

ORIGIN_DIR=`pwd`

cd /home/jieyi/Ani/code/VideoPreprocess

if [ ! -d $DATA_DIR/$NAME/images ];then
    mkdir $DATA_DIR/$NAME/images
fi

echo "step 1. extract images"
python extract_image.py --data_path $DATA_DIR/$NAME/video.mp4 --save_path $DATA_DIR/$NAME/images --image_num 10

# extract the location embedding
echo "step 2. extract the location embedding"
cd /home/jieyi/Ani/code/SceneClassification/Scene-Classification
python LocationEmbedding.py --data_path $DATA_DIR/$NAME/images --save_path $DATA_DIR/$NAME


# extract the activity embeddings
echo "step 3. extract the activity embedding"
cd /home/jieyi/Ani/data/EJShibaVoice/scripts
python ActivityEmbedding.py --data_path $DATA_DIR/$NAME/images --save_path $DATA_DIR/$NAME


