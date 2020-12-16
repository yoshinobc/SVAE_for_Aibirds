import sys
import os
import argparse
import json

import numpy as np
from chainer import Variable
from tqdm import tqdm

from traintest import train, test

from generator.model_vae import VAE


sys.path.append("../../")
from my_converter import txt2xml, xml2txt

class Args():
    def __init__(self, train=True):
        self.dataname = "aibirds_word"
        self.train_source = "datasets/{}_train.txt".format(self.dataname)
        self.test_source = "datasets/{}_test.txt".format(self.dataname)
        self.epoch = 401
        self.n_vocab = 1505
        #self.embed = 50
        self.hidden = 300
        self.n_latent = 50
        self.layer = 1
        self.batchsize = 20
        self.test_batchsize = 20
        self.sample_size = 10
        self.kl_zero_epoch = 201
        self.dropout = 0.3
        self.noise = 0.1
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 5
        self.result_dir = "aibirds_word_{}_{}_{}_{}_{}_{}".format(self.hidden, self.n_latent, self.batchsize, self.dropout, self.gradclip, self.noise)

def sampleTrain():
    args = Args(True)
    encdec = VAE(args)
    os.mkdir(args.result_dir)
    with open(args.result_dir+"/parameters.json", "w") as f:
        f.write(json.dumps(args.__dict__))
    os.mkdir(args.result_dir+"/models")
    train(args, encdec, args.result_dir+"/models/{}_{}_{}_{}")

def testDec(args, encdec, batch_size):
    tenti = encdec.predict(batch_size, randFlag = True)
    encdec.dec.reset_state()
    return tenti

def sampleTest():
    args = Args()
    dir_name = input("input dir name >> ")
    with open(dir_name+"/parameters.json", "r") as f:
        jsn = json.load(f)
    for jsn_key in jsn:
        setattr(args, jsn_key, jsn[jsn_key])
    #args.gpu = -1
    encdec = VAE(args)
    epoch = input("input epoch >> ")
    n_latent = input("input n_latent >> ") 
    model_name = "./{}/models/aibirds_word_aibirds_word_{}_{}".format(dir_name, epoch, n_latent)
    encdec = test(args, encdec, model_name)
    deconverter = txt2xml.txt2xml()
    os.makedirs("make_levels",exist_ok=True)
    sample_size = int(input("input sample_size >> "))
    uni_gram = {}
    bi_gram = {}
    for i in tqdm(range(sample_size)):
        tenti = testDec(args, encdec, 1)
        uni_gram, bi_gram = bi_uni(uni_gram, bi_gram, tenti)
        text = deconverter.vector2xml(tenti[0])
        with open("make_levels/level-" + str(i) + ".xml", "w") as f:
            f.write(text)
    print("uni", len(sorted(uni_gram.items(), key=lambda x: x[1])))
    print("bi", len(sorted(bi_gram.items(), key=lambda x: x[1]))) 

def bi_uni(uni_gram, bi_gram, tenti):
    for data in tenti:
        old_line = ""
        for word in data:
            word = "".join(word)
            if word in uni_gram.keys():
                 uni_gram[word] += 1
            else:
                uni_gram[word] = 1
            concat_line = old_line + word
            if concat_line in bi_gram.keys():
                bi_gram[concat_line] += 1
            else:
                bi_gram[concat_line] = 1
            old_line = word
    return uni_gram, bi_gram
"""
def testSentAdd(args, encdec):
    conv = xml2vector.xml2vector()
    test_data = conv.maketest()
    tenti = testAdd(args, encdec, test_data)
    vec2xml = vector2xml.vector2xml()
    for i, name in enumerate(tenti):
        text = vec2xml.vector2xml(name)
        with open("make_levels/level-"+str(i)+".xml", mode="w") as f:
            f.write(text)


def sampleTest():
    def save_xml(vec2xml, z, args, encdec, batch_size, index):
        tenti = testDec(z, args, encdec, batch_size)
        index = str(index)
        os.makedirs("make_levels/"+index,exist_ok=True)
        for i, name in enumerate(tenti[:10]):
            text = vec2xml.vector2xml(name)
            with open("make_levels/"+index+"/level-"+str(i)+".xml",mode="w") as f:
                f.write(text)

    args = Args(False)
    for e_i in [100]:
        encdec = VAE(args)
        model_name = "./{}/model/vae_biconcat_kl_{}_{}_l{}.npz".format(
            args.dataname, args.dataname, e_i, args.n_latent)
        encdec = test(args, encdec, model_name)

        vec2xml = vector2xml.vector2xml()
        z_list = []
        np.random.seed(seed=11)
        z1 = np.random.normal(0, 1, (args.batchsize, args.n_latent)).astype(np.float32)
        z2 = np.random.normal(0, 1, (args.batchsize, args.n_latent)).astype(np.float32)
        for i, ri in enumerate(np.linspace(0.0, 1.0, num=10)):
            z = z1 * (1-ri) + z2 * ri
            save_xml(vec2xml,z,args,encdec, args.batchsize, i)
        import pickle
        pickle.dump(z_list, open("z_list.pkl","wb"))
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--train",
                        help="train mode",
                        action="store_true")
    args = parser.parse_args()

    if args.train:
        sampleTrain()
    else:
        sampleTest()

