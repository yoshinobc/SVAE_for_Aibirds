from chainer import functions as F
from chainer import Variable


"""
def testDec(z, args, encdec, batchsize):
    z = Variable(z)
    tenti = encdec.predict(batchsize, z=z,randFlag=True)
    encdec.dec.reset_state()
    return tenti


def testAdd(args, encdec, sent_arr, times=10):
    mu_arr, var_arr = vectorize(args, encdec, sent_arr)
    mu_arr = [mu for mu, var in zip(mu_arr, var_arr)]
    ratio = 1.0/times
    vec_arr = [mu_arr[1]]+[ri*ratio*mu_arr[0] +
                           (1.0 - ri * ratio) * mu_arr[1] for ri in range(times + 1)] + [mu_arr[0]]
    print(vec_arr[0].shape)
    print(len(vec_arr))
    #test = F.reshape(F.concat(vec_arr), (len(vec_arr), args.n_latent))
    # print(test[0])
    # print(len(test))
    # print(len(test[0]))
    # print(F.reshape(
    #    F.concat(vec_arr), (len(vec_arr), args.n_latent))[0])
    # print(len(F.reshape(
    #    F.concat(vec_arr), (len(vec_arr), args.n_latent))[0]))
    exit()
    if len(mu_arr) > 1:
        tenti = encdec.predict(len(vec_arr), randFlag=False, z=F.reshape(
            F.concat(vec_arr), (len(vec_arr), args.n_latent)))
    encdec.dec.reset_state()
    return tenti
"""

def vectorize(args, encdec, sent_arr):
    tt_batch = [[encdec.vocab.stoi(char) for char in word_arr.split(
        "  ")] for word_arr in sent_arr]
    mu_arr, var_arr = encdec.encode(tt_batch)

    mu_arr = mu_arr[0]
    mu_arr = F.split_axis(mu_arr, len(sent_arr), axis=0)
    var_arr = var_arr[0]
    var_arr = F.split_axis(var_arr, len(sent_arr), axis=0)

    return mu_arr, var_arr


