from chainer import serializers, optimizers
from chainer import functions as F
import chainer
import numpy as np

import os
import sys

sys.path.append("../../")
from my_converter import txt2xml, xml2txt


def train(args, encdec, model_name_base):   
    #encdec.loadModel(model_name_base, args)
    if args.gpu >= 0:
        encdec.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(encdec)

    encdec.embed.W.update_rule.enabled = False
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    with open(args.result_dir+"/loss_train.txt", "w") as f:
        f.write("loss_train, rec_loss, kl_loss\n")
    with open(args.result_dir+"/loss_test.txt", "w") as f:
        f.write("loss_test, rec_loss, kl_loss\n")

    vectors = open(args.train_source, "r")
    deconverter = txt2xml.txt2xml()
    os.makedirs(args.result_dir+"/train/sample_levels", exist_ok=True)
    for i, vector in enumerate(vectors):
        if i == args.batchsize:
            break
        text = deconverter.txt2xml(vector)
        with open(args.result_dir + "/train/sample_levels/level-"+str(i)+".xml","w") as f:
            f.write(text)

    vectors = open(args.test_source, "r")
    deconverter = txt2xml.txt2xml()
    os.makedirs(args.result_dir+"/test/sample_levels", exist_ok=True)
    for i, vector in enumerate(vectors):
        text = deconverter.txt2xml(vector)
        with open(args.result_dir + "/test/sample_levels/level-"+str(i)+".xml", "w") as f:
            f.write(text)
    print(encdec.epoch_now, args.epoch)
    for e_i in range(encdec.epoch_now, args.epoch):
        encdec.setEpochNow(e_i)
        loss_sum = 0
        rec_loss_sum = 0
        kl_loss_sum = 0
        count = 0
        for tupl in encdec.getBatchGen(args):
            count += 1
            loss, rec_loss, _ = encdec(tupl)
            loss_sum += loss.data
            rec_loss_sum += float(rec_loss)
            kl_loss_sum += (loss.data - rec_loss)
            encdec.cleargrads()
            if np.isnan(loss.data).any():
                exit()
                pass
            else:
                loss.backward()
                optimizer.update()
        print("epoch {}:loss_sum: {}".format(e_i, loss_sum/count))
        print("epoch {}:rec_loss_mean: {}".format(e_i, rec_loss_sum/count))
        print("epoch {}:kl_loss_mean: {}".format(e_i, kl_loss_sum / count))

        with open(args.result_dir+"/loss_train.txt", "a") as f:
            f.write(str(loss_sum/count) + "," + str(rec_loss_sum /
                                                    count) + "," + str(kl_loss_sum/count) + "\n")
        count = 0
        for tupl in encdec.getBatchGen_test(args):
            count += 1
            loss, rec_loss, _ = encdec(tupl, True)
            loss_sum += loss.data
            rec_loss_sum += float(rec_loss)
            kl_loss_sum += (loss.data - rec_loss)
            encdec.cleargrads()
        print("test epoch {}:loss_sum: {}".format(e_i, loss_sum/count))
        print("test epoch {}:rec_loss_mean: {}".format(e_i, rec_loss_sum/count))
        print("test epoch {}:kl_loss_mean: {}".format(e_i, kl_loss_sum/count))
        with open(args.result_dir+"/loss_test.txt", "a") as f:
            f.write(str(loss_sum/count) + "," + str(rec_loss_sum /
                                                    count) + "," + str(kl_loss_sum/count) + "\n")
        if e_i % 5 == 0:
            os.makedirs(args.result_dir+"/train/sample_levels/" +
                        str(e_i), exist_ok=True)
            os.makedirs(args.result_dir+"/test/sample_levels/" + str(e_i), exist_ok=True)
            i = 0
            for tupl in encdec.getBatchGen(args, False):
                mu_arr, var_arr = encdec.encode(tupl)
                z = F.gaussian(mu_arr[0], var_arr[0])
                tenti = encdec.predict(args.batchsize, z=z)
                encdec.dec.reset_state()
                for _, name in enumerate(tenti):
                    text = deconverter.vector2xml(name)
                    with open(args.result_dir+"/train/sample_levels/"+str(e_i)+"/level-"+str(i)+".xml","w") as f:
                        f.write(text)
                break
            
            i = 0
            for tupl in encdec.getBatchGen_test(args, False):
                mu_arr, var_arr = encdec.encode(tupl)
                z = F.gaussian(mu_arr[0], var_arr[0])
                tenti = encdec.predict(args.batchsize, z=z)
                encdec.dec.reset_state()
                for _, name in enumerate(tenti):
                    text = deconverter.vector2xml(name)
                    with open(args.result_dir+"/test/sample_levels/"+str(e_i)+"/level-"+str(i)+".xml", "w") as f:
                        f.write(text)
                    i += 1
            
            model_name = model_name_base.format(
                args.dataname, args.dataname, e_i, args.n_latent)
            serializers.save_npz(model_name, encdec)


def test(args, encdec, model_name, categ_arr=[], predictFlag=False):
    serializers.load_npz(model_name, encdec)
    if args.gpu >= 0:
        encdec.to_gpu()
    encdec.setBatchSize(args.batchsize)

    if predictFlag:
        encdec.predict(args.batchsize, randFlag=False)
    return encdec
