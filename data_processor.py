#!/usr/bin/env python
#-*- coding:utf-8 -*-
# file name: data_processor.py
# file description:
# created at:1/11/2021 1:49 PM

import pickle as pkl

def create_one(data_sbto_src, data_sbt_graph, set):
    if set != "train" and set != "val" and set != "test":
        raise Exception("No train, val, and test available !")
    src = []
    ast = []
    tgt = []
    for item1, item2 in zip(data_sbto_src[set], data_sbt_graph[set]):
        if item1[-1].strip().lower() != item2[-1].strip().lower():
            raise Exception("alignment error!")
        ast.append(item2[1])
        src.append(item1[1])
        tgt.append(item2[-1])
    with open("./samples/smart_contracts/{}/{}.spl.src".format(set, set), "w", encoding="utf-8") as fw:
        for line in src:
            fw.write(line+"\n")
    with open("./samples/smart_contracts/{}/{}.ast.src".format(set, set), "w", encoding="utf-8") as fw:
        for line in ast:
            fw.write(line+"\n")
    with open("./samples/smart_contracts/{}/{}.txt.tgt".format(set, set), "w", encoding="utf-8") as fw:
        for line in tgt:
            fw.write(line+"\n")

def create_dataset():
    with open("./samples/smart_contracts/dataset_train_val_test_uniq_sbto_src.pkl", "rb") as fr:
        data1 = pkl.load(fr)
    with open("./samples/smart_contracts/dataset_train_val_test_uniq_sbt_graph.pkl", "rb") as fr:
        data2 = pkl.load(fr)
    print(data1.keys())
    print(data2.keys())
    print("creating dataset for train...")
    create_one(data1, data2, "train")
    print("creating dataset for valid...")
    create_one(data1, data2, "val")
    print("creating dataset for test...")
    create_one(data1, data2, "test")


if __name__ == "__main__":
    create_dataset()

