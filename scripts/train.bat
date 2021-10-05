python train.py --train-data-src ../samples/python/train/train.spl.src --train-data-tgt ../samples/python/train/train.txt.tgt
                                        --dev-data-src ../samples/python/valid/valid.spl.src --dev-data-tgt ../samples/python/valid/valid.txt.tgt
                                        --vocab ./dataset/mix_vocab.json
                                        --cuda
                                        --input-feed \
                                        --share-embed \
                                        --mix-vocab \
                                        --dropout 0.2 \
                                        --use-pre-embed \
                                        --freeze-pre-embed \
                                        --vocab-embed ${ds_dir}/mix_vocab_embeddings.pkl \
                                        --model-class ${model_class} \
                                        --log-dir ${dir} \
                                        --save-to ${model_path}
