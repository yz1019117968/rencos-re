## Re-implementation for Rencos
This is a re-implementation for Rencos. The original repo please refers to [here](https://github.com/zhangj111/rencos). 
### Dependency
```cmd
python == 3.6
pytorch == 1.9.0
nltk == 3.6.3
rouge == 1.0.1 ref: https://github.com/pltrdy/rouge
docopt == 0.6.2
nlg-eval == 2.3 ref: https://github.com/Maluuba/nlg-eval
```

### Quick Start  
Execute the following cmd step by step to use rencos.
- Build Vocab.
```cmd
cd scripts
vocab.bat
```
- Train Base NMT.
```cmd
cd scripts
train.bat
```
- Infer for base NMT.
```cmd
cd scripts
infer.bat
```
- Evaluation for base NMT.
```cmd
cd scripts
eval.bat
```
- Retrieve the most similar sources based on structure and semantic, respectively.
```cmd
cd scripts
retrieve.bat
```
- Normalize distance scores between queries and sources.
```cmd
cd scripts
normalize.bat
```
- Infer for rencos
```cmd
cd scripts
infer_rencos.bat
```
- Evaluate for rencos
```cmd
cd scripts
eval_rencos.bat
```

### Results for smart contracts
| Structure | S-BLEU| C-BLEU | ROUGE | METEOR | CIDER |
| ---- | ---- | ---- | ---- | ---- | ---- |
| NMT+Beam | 0.322598 | 0.352737 | 0.436517 | 0.470094 | 3.201473 |
| NMT+IR+Beam | 0.325910 | 0.355818 | 0.434394 | 0.463494 | 3.227261 |
| IR.0 | 0.424972 | 0.455662 | 0.523240 | 0.554297 | 4.229340 |
| IR.1 | 0.466490 | 0.498686 | 0.556926 | 0.584497 | 4.619538 |


