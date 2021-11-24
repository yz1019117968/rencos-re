@ echo off
set lang=smart_contracts
set model_path=../models/20211123-180054/model.bin
set vocab_file=../samples/%lang%/dic/%lang%_dic.json
set train_set_src=../samples/%lang%/train/train.spl.src
set train_set_tgt=../samples/%lang%/train/train.txt.tgt
set test_set_src=../samples/%lang%/test/test.spl.src
set test_set_tgt=../samples/%lang%/test/test.txt.tgt
set query_out_path=../samples/%lang%/test/test.vec.pkl
set source_out_path=../samples/%lang%/train/train.vec.pkl
set simi_id_out=../samples/%lang%/test/id_score
set test_ref_src_1=../samples/%lang%/test/test.ref.src.1
set test_ref_tgt_1=../samples/%lang%/test/test.ref.tgt.1

python syntax.py %lang%

python ../semantic.py %model_path% %vocab_file% %train_set_src% %train_set_tgt% %test_set_src% %test_set_tgt% ^
                      %query_out_path% %source_out_path% %simi_id_out% %test_ref_src_1% %test_ref_tgt_1%
