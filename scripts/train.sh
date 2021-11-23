lang=smart_contracts
model_folder=$(date "+%Y%m%d-%H%M%S")
model_path=../models/${model_folder}/model.bin
log_dir=../models/${model_folder}/log/
python ../train.py --clip-grad -1 \
                   --save-to ${model_path} \
                   --log-dir ${log_dir} \
                    ../samples/${lang}/train/train.spl.src  \
                    ../samples/${lang}/train/train.txt.tgt \
                    ../samples/${lang}/val/val.spl.src \
                    ../samples/${lang}/val/val.txt.tgt \
                    ../samples/${lang}/dic/${lang}_dic.json

