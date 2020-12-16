import sys
import os
import argparse
import json

from tqdm import tqdm
import numpy as np
from chainer import Variable

from traintest import train, test

from generator.model_vae import VAE


sys.path.append("../../")
from my_converter import txt2xml_sparse, xml2txt_sparse

class Args():
    def __init__(self, train=True):
        self.dataname = "aibirds_word"
        self.train_source = "datasets/{}_train.txt".format(self.dataname)
        self.test_source = "datasets/{}_test.txt".format(self.dataname)
        self.epoch = 401
        self.n_vocab = 61
        self.embed = 50
        self.hidden = 400
        self.n_latent = 10
        self.layer = 1
        self.batchsize = 10
        self.test_batchsize = 10
        self.sample_size = 1
        self.kl_zero_epoch = 201
        self.dropout = 0.3
        self.noise = 0.1
        if train:
            self.gpu = 0
        else:
            self.gpu = -1
        self.gradclip = 3
        self.result_dir = "aibirds_word_{}_{}_{}_{}_{}_{}_{}".format(self.embed, self.hidden, self.n_latent, self.batchsize, self.dropout, self.gradclip, self.noise)

def sampleTrain():
    args = Args(True)
    encdec = VAE(args)
    os.mkdir(args.result_dir)
    with open(args.result_dir+"/parameters.json", "w") as f:
        f.write(json.dumps(args.__dict__))
    os.mkdir(args.result_dir+"/models")
    train(args, encdec, args.result_dir+"/models/{}_{}_{}_{}")


def testDec(args, encdec, batch_size):
    tenti = encdec.predict(batch_size)
    encdec.dec.reset_state()
    return tenti

def char2word(tenti):
    count = 0
    word = []
    sentence = []
    for char in tenti:
        if count == 94:
            sentence.append(word)
            word = []
            count = 0
        word.append(char)
        count += 1
    return [sentence]

        
def sampleTest():
    args = Args()
    dir_name = input("input dir name >> ")
    with open(dir_name+"/parameters.json", "r") as f:
        jsn = json.load(f)
    for jsn_key in jsn:
        setattr(args, jsn_key, jsn[jsn_key])
    args.gpu = -1
    encdec = VAE(args)
    epoch = input("input epoch >> ")
    model_name = "./{}/models/aibirds_word_aibirds_word_{}_10".format(dir_name, epoch)
    encdec = test(args, encdec, model_name)
    deconverter = txt2xml_sparse.txt2xml()
    os.makedirs("make_levels",exist_ok=True)
    sample_size = int(input("input sample_size >> "))
    for i in tqdm(range(sample_size)):
        tenti = testDec(args, encdec, 1)
        tenti = char2word(tenti)
        text = deconverter.vector2xml(tenti[0], is_True=True)
        with open("make_levels/level-" + str(i) + ".xml", "w") as f:
            f.write(text)

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
        #pass

