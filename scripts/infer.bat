@ echo off
set model_path=../models/20211010-210944/model.bin
set test_set_src=../samples/python/test/test.spl.src
set test_set_tgt=../samples/python/test/test.txt.tgt
set output_file=../models/20211010-210944/results.jsonl
set output_file_obs=../models/20211010-210944/results.txt

python ../infer.py  %model_path% ^
               %test_set_src% ^
               %test_set_tgt% ^
               %output_file% ^
               %output_file_obs%
