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
    uni_gram = {}
    bi_gram = {}
    for i in tqdm(range(sample_size)):
        tenti = testDec(args, encdec, 1)
        #print(tenti)
        tenti = char2word(tenti)
        uni_gram, bi_gram = bi_uni(uni_gram, bi_gram, tenti)
        #print(len(tenti), len(tenti[0]), len(tenti[0][0]))
        text = deconverter.vector2xml(tenti[0], is_True=True)
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
        #pass

