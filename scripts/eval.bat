set lang=smart_contracts

cd ..
python ./eval.py ./samples/%lang%/test/test.spl.src ^
                  ./samples/%lang%/test/test.txt.tgt ^
                  ./models/20211123-180054/results.jsonl
