from chainer import cuda
from chainer import link
import numpy as np

def predictRandom(prob):
    probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    return index