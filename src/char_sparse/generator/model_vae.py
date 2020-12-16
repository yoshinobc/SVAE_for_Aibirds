import chainer.links as L
from chainer import Variable
import random
import generator.utils.generators as gens
import chainer.functions as F
import numpy as np

from generator.model_common import VAECommon, LSTM

class VAE(VAECommon):

    def __init__(self, args):
        self.setArgs(args)
        super(VAE, self).__init__(
            embed=L.EmbedID(self.n_vocab, self.n_embed),
            # VAEenc
            enc_f=LSTM(self.n_layers, self.n_embed,
                       self.out_size, dropout=self.drop_ratio),
            enc_b=LSTM(self.n_layers, self.n_embed,
                       self.out_size, dropout=self.drop_ratio),

            le2_mu=L.Linear(4*self.out_size, self.n_latent),
            le2_ln_var=L.Linear(4*self.out_size, self.n_latent),
            # VAEdec
            ld_h=L.Linear(self.n_latent, 2*self.out_size),
            ld_c=L.Linear(self.n_latent, 2*self.out_size),

            dec=LSTM(self.n_layers, self.n_embed, 2 * \
                     self.out_size, dropout=self.drop_ratio),
            h2w=L.Linear(2*self.out_size, self.n_vocab),
        )

    def setArgs(self, args):
        if args.gpu >= 0:
            global np
            import cupy as np
        super().setArgs(args)

    def getBatchGen(self, args, is_shuffle=True):
        tt_now_list = [[self.vocab.stoi(char) for char in char_arr]
                       for char_arr in gens.word_list(args.train_source)]
        ind_arr = list(range(len(tt_now_list)))
        if is_shuffle:
            random.shuffle(ind_arr)
        tt_now = (tt_now_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.batchsize)
        for tt in tt_gen:
            yield tt

    def getBatchGen_test(self, args, is_shuffle=True):
        tt_now_list = [[self.vocab.stoi(char) for char in char_arr]
                       for char_arr in gens.word_list(args.test_source)]
        ind_arr = list(range(len(tt_now_list)))
        if is_shuffle:
            random.shuffle(ind_arr)
        tt_now = (tt_now_list[ind] for ind in ind_arr)
        tt_gen = gens.batch(tt_now, args.test_batchsize)
        for tt in tt_gen:
            yield tt

    def __call__(self, xs, is_fukugen=False):
        print(xs)
        mu_arr, var_arr = self.encode(xs)
        t = [[1]+x for x in xs]  # 1は<s>を指す。decには<s>から入れる。</s>まで予測する。
        loss = None
        rec_loss = None
        for mu, var in zip(mu_arr, var_arr):
            loss, rec_loss, ys_w = self.calcLoss(t, mu, var)
        return loss, rec_loss, ys_w

    def calcLoss(self, t, mu, ln_var):
        k = self.sample_size
        kl_zero_epoch = self.kl_zero_epoch
        loss = None
        t_pred = [t_e[1:]+[2] for t_e in t]
        t_pred = [np.asarray(tp_e, dtype=np.int32) for tp_e in t_pred]
        t = self.denoiseInput(t)
        t_vec = self.makeEmbedBatch(t)
        for _ in range(k):
            z = F.gaussian(mu, ln_var)
            if loss is None:
                loss, ys_w = self.decode(
                    z, t_vec, t_pred)
                loss /= (k * self.batch_size)
            elif loss is not None:
                loss_, ys_w = self.decode(
                    z, t_vec, t_pred)
                loss_ /= (k * self.batch_size)
                loss += loss_
        C = 0.01 * (self.epoch_now-kl_zero_epoch) / self.epoch
        rec_loss = loss
        if self.epoch_now > kl_zero_epoch:
            loss += C * F.gaussian_kl_divergence(mu, ln_var) / self.batch_size
        return loss, rec_loss.data, ys_w

    def decode(self, z, t_vec, t_pred):
        self.dec.hx = F.reshape(
            self.ld_h(z), (1, self.batch_size, 2*self.out_size))
        self.dec.cx = F.reshape(
            self.ld_c(z), (1, self.batch_size, 2 * self.out_size))
        loss, ys_w = super().decode(t_vec, t_pred)
        return loss, ys_w

    def predict(self, batch_size, z=None, randFlag=False):
        if z is None:
            z = Variable(np.random.normal(
                0, 1, (batch_size, self.n_latent)).astype(np.float32))
        self.dec.hx = F.reshape(self.ld_h(z), (1, batch_size, 2*self.out_size))
        self.dec.cx = F.reshape(self.ld_c(z), (1, batch_size, 2*self.out_size))
        tenti = super().predict(batch_size, randFlag)
        return tenti
