set model_folder=%date:~10,4%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%%time:~6,2%
set model_path=../models/%model_folder%/model.bin
set log_dir=../models/%model_folder%/log/
python ../train.py --clip-grad -1 ^
                   --save-to %model_path% ^
                   --log-dir %log_dir% ^
                    ../samples/python/train/train.spl.src  ^
                    ../samples/python/train/train.txt.tgt ^
                    ../samples/python/valid/valid.spl.src ^
                    ../samples/python/valid/valid.txt.tgt ^
                    ../samples/python/dic/python_dic.json


