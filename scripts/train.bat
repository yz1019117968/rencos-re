set lang=smart_contracts
set model_folder=%date:~10,4%%date:~7,2%%date:~4,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set model_path=../models/%model_folder%/model.bin
set log_dir=../models/%model_folder%/log/
python ../train.py --clip-grad -1 ^
                   --save-to %model_path% ^
                   --log-dir %log_dir% ^
                    ../samples/%lang%/train/train.spl.src  ^
                    ../samples/%lang%/train/train.txt.tgt ^
                    ../samples/%lang%/val/val.spl.src ^
                    ../samples/%lang%/val/val.txt.tgt ^
                    ../samples/%lang%/dic/%lang%_dic.json


