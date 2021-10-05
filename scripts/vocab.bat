python ../vocab.py --train-set-src ../samples/python/train/train.spl.src ^
         --train-set-tgt ../samples/python/train/train.txt.tgt ^
         --size-src 50000 --size-tgt 50000 ^
         --freq-cutoff 2 --vocab-class Vocab ^
         ../samples/python/dic/python_dic.json
