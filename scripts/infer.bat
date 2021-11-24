@ echo off
set lang=smart_contracts
set model_path=./models/20211123-180054/model.bin
set test_set_src=./samples/%lang%/test/test.spl.src
set test_set_tgt=./samples/%lang%/test/test.txt.tgt
set output_file=./models/20211123-180054/results.jsonl
set output_file_obs=./models/20211123-180054/results.txt

cd ..
python ./infer.py  %model_path% ^
               %test_set_src% ^
               %test_set_tgt% ^
               %output_file% ^
               %output_file_obs%
