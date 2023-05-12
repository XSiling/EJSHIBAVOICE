#!/bin/bash
#!/bin/bash
source /opt/anaconda3/bin/activate

DATA_DIR=$1 #/home/jieyi/Ani/data/ShibaLang/ShibaInuKohachannel
NAME=$2
ORIGIN_DIR=`pwd`
echo "/home/jieyi/Ani/demo/acmmm_demo/run.sh"
# if [ ! -d /home/jieyi/${NAME}/ ];then
#     mkdir /home/jieyi/${NAME}/
# fi

echo /home/jieyi/Ani/demo/acmmm_demo/static/upload/$NAME.wav
# echo /home/jieyi/$NAME/$NAME.wav
# cp /home/jieyi/Ani/demo/acmmm_demo/static/upload/$NAME.wav /home/jieyi/$NAME/$NAME.wav


# 1. raw panns to get "sentence"
echo "step 1. raw panns to get sentence"
cd /home/jieyi/Ani/AudioTagging/audioset_tagging_cnn
LOG_PATH=${DATA_DIR}/log.txt


echo "python pytorch/process_shiba.py sound_event_detection --log_file ${LOG_PATH} --model_type Cnn14_DecisionLevelMax --checkpoint_path /home/jieyi/panns_data/Cnn14_DecisionLevelMax.pth --file_path /home/jieyi/${NAME}/ --wav_path /home/jieyi/${NAME}/ --cuda"
python pytorch/process_shiba_acmmm.py sound_event_detection --log_file ${LOG_PATH} --model_type Cnn14_DecisionLevelMax --checkpoint_path /home/jieyi/panns_data/Cnn14_DecisionLevelMax.pth --file_path /home/jieyi/Ani/demo/acmmm_demo/static/data/${NAME}/ --wav_path /home/jieyi/Ani/demo/acmmm_demo/static/data/${NAME}/ --cuda

# 2. process the result of panns
echo "step 2. process the result of panns"
cd $ORIGIN_DIR

SAVE_PATH=${DATA_DIR}/sentences/

if [ ! -d ${SAVE_PATH} ];then
    mkdir ${SAVE_PATH}
fi



python process_pannsresult.py --log_path ${LOG_PATH} --save_path ${SAVE_PATH} --audio_path /home/jieyi/Ani/demo/acmmm_demo/static/data/${NAME}/

# 4. remove those noise
echo "step 3. remove noise"
cd /home/jieyi/Ani/AudioTagging/audioset_tagging_cnn
python pytorch/process_noiseremover.py sound_event_detection --model_type Cnn14_DecisionLevelMax --checkpoint_path /home/jieyi/panns_data/Cnn14_DecisionLevelMax.pth --file_path ${SAVE_PATH} --cuda


# 3. automatically generate sentence->word
echo "step 4. automatically generate sentence to word"
cd /home/jieyi/Ani/code/CutRawClips/
OUT_DIR_AUTO=/home/jieyi/Ani/code/CutRawClips/result_${NAME}.txt
SAVE_PATH_WORD=${DATA_DIR}/words/
if [ ! -d ${SAVE_PATH_WORD} ];then
    mkdir ${SAVE_PATH_WORD}
fi 

conda activate maskrcnn_benchmark
echo "inference.py --out_dir ${OUT_DIR_AUTO} --audio_path ${SAVE_PATH}"
python inference.py --out_dir ${OUT_DIR_AUTO} --audio_path ${SAVE_PATH}
echo "processresult.py --audio_path ${SAVE_PATH} --save_path ${SAVE_PATH_WORD} --result_file ${OUT_DIR_AUTO}"
python processresult.py --audio_path ${SAVE_PATH} --save_path ${SAVE_PATH_WORD} --result_file ${OUT_DIR_AUTO}
conda deactivate

# 4. zip all the audio files


