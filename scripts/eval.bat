set lang=smart_contracts
rem ./models/20211123-180054/results.jsonl ^
cd ..
python ./eval.py ./samples/%lang%/test/test.spl.src ^
                  ./samples/%lang%/test/test.txt.tgt ^
                  ./samples/%lang%/test/test.ref.tgt.1 ^
                  --source ir
