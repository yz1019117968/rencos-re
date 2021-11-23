set lang=smart_contracts
python ../vocab.py ../samples/%lang%/train/train.spl.src ^
         ../samples/%lang%/train/train.txt.tgt ^
         ../samples/%lang%/dic/%lang%_dic.json
