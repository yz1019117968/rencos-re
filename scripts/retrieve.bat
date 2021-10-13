@ echo off
set model_path=../models/20211010-210944/model.bin
set vocab_file=../samples/python/dic/python_dic.json
set train_set_src=../samples/python/train/train.spl.src
set train_set_tgt=../samples/python/train/train.txt.tgt
set test_set_src=../samples/python/test/test.spl.src
set test_set_tgt=../samples/python/test/test.txt.tgt
rem python syntax python
python ../semantic.py %model_path% %vocab_file% %train_set_src% %train_set_tgt% %test_set_src% %test_set_tgt%
