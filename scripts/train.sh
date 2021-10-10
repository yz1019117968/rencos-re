model_folder=$(date "+%Y%m%d-%H%M%S")
model_path=../models/${model_folder}/model.bin
log_dir=../models/${model_folder}/log/
python ../train.py --clip-grad -1 \
                   --save-to ${model_path} \
                   --log-dir ${log_dir} \
                    ../samples/python/train/train.spl.src  \
                    ../samples/python/train/train.txt.tgt \
                    ../samples/python/valid/valid.spl.src \
                    ../samples/python/valid/valid.txt.tgt \
                    ../samples/python/dic/python_dic.json

