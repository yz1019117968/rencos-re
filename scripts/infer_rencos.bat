@ echo off
set lang=smart_contracts
set model_path=../models/20211123-180054/model.bin
set test_set_src=../samples/%lang%/test/test.spl.src
set test_set_tgt=../samples/%lang%/test/test.txt.tgt
set output_file=../models/20211123-180054/results_rencos.jsonl
set output_file_obs=../models/20211123-180054/results_rencos.txt
set test_ref_src_0=../samples/%lang%/test/test.ref.src.0
set test_ref_src_1=../samples/%lang%/test/test.ref.src.1
set test_ref_tgt_0=../samples/%lang%/test/test.ref.tgt.0
set test_ref_tgt_1=../samples/%lang%/test/test.ref.tgt.1
set prs_0=../samples/%lang%/test/prs.0
set prs_1=../samples/%lang%/test/prs.1

python ../infer.py  %model_path% ^
               %test_set_src% ^
               %test_set_tgt% ^
               %output_file% ^
               %output_file_obs% ^
               --rencos True ^
               --test-ref-src-0 %test_ref_src_0% ^
               --test-ref-src-1 %test_ref_src_1% ^
               --test-ref-tgt-0 %test_ref_tgt_0% ^
               --test-ref-tgt-1 %test_ref_tgt_1% ^
               --prs-0 %prs_0% --prs-1 %prs_1% ^
               --lambda 3
