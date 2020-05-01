

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io


# change the directory for the other dataset
filename = "Data/Alice.txt"
raw_text = open(filename).read()
text = raw_text.lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars), 'total:', len(text))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


xtrain = np.zeros((len(text), len(chars)))

for i, char in enumerate(text):
    xtrain[i, char_indices[char]] = 1.0


INPUT = len(chars)

HIDDEN = 200
OUTPUT = len(chars)

INPUT += HIDDEN

ALPHA = 0.005
Beta1 = 0.9
Beta2 = 0.999
Epsilon = 1e-8

BATCH_NUM = 500
Time = BATCH_NUM
ITER_NUM = 10
theta = 0.1

LOG = 1
TextLOG = 50
errors = []  # to plot learning curve of cross entropy

wf = np.random.randn(INPUT, HIDDEN) / np.sqrt(INPUT / 2)
wi = np.random.randn(INPUT, HIDDEN) / np.sqrt(INPUT / 2)
wc = np.random.randn(INPUT, HIDDEN) / np.sqrt(INPUT / 2)
wo = np.random.randn(INPUT, HIDDEN) / np.sqrt(INPUT / 2)
wy = np.random.randn(HIDDEN, OUTPUT) / np.sqrt(HIDDEN / 2)

bf = np.zeros(HIDDEN)
bi = np.zeros(HIDDEN)
bc = np.zeros(HIDDEN)
bo = np.zeros(HIDDEN)
by = np.zeros(OUTPUT)

dwf = np.zeros_like(wf)
dwi = np.zeros_like(wi)
dwc = np.zeros_like(wc)
dwo = np.zeros_like(wo)
dwy = np.zeros_like(wy)

dbf = np.zeros_like(bf)
dbi = np.zeros_like(bi)
dbc = np.zeros_like(bc)
dbo = np.zeros_like(bo)
dby = np.zeros_like(by)


def softmax(arr):
    c = np.clip(arr, -700, 700)  # float64 maximum expotentiable value
    e = np.exp(c)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(out, label):
    entropy = label * np.log(out + 1e-6)  # to prevent log value overflow
    return -np.sum(entropy, axis=1, keepdims=True)

def spike(arr):
    # print(arr)
    C = 1 * (arr > theta)
    Dif = arr - theta
    return C, Dif


def deriv_spike(out):
    return (1 / np.sqrt(4*np.pi)) * np.exp(-(out ** 2)/4)

def deriv_Tanhspike(out):
    return (1 / np.sqrt(0.3*np.pi)) * np.exp(-(out ** 2)/0.3)

mwf, mwi, mwc, mwo, mwy = 0, 0, 0, 0, 0
vwf, vwi, vwc, vwo, vwy = 0, 0, 0, 0, 0

mbf, mbi, mbc, mbo, mby = 0, 0, 0, 0, 0
vbf, vbi, vbc, vbo, vby = 0, 0, 0, 0, 0

def LSTM_Cell(input_val):
    caches = []
    states = []

    states.append([np.zeros([1, HIDDEN]), np.zeros([1, HIDDEN])])
    Time = input_val.shape[0]

    for t in range(Time):

        splitedInput = input_val[t].reshape(1, -1)
        c_prev, h_prev = states[-1]
        x = np.column_stack([splitedInput, h_prev])

        hf, Dhf = spike(np.dot(x, wf) + bf)
        hi, Dhi = spike(np.dot(x, wi) + bi)
        ho, Dho = spike(np.dot(x, wo) + bo)
        hc, Dhc = spike(np.dot(x, wc) + bc)
        #print(hf * c_prev + hi * hc)
        c, Dc = spike(hf * c_prev + hi * hc)

        h = ho * c
        y = np.dot(h, wy) + by
        pred = softmax(y)

        states.append([c, h])
        caches.append([x, hf, Dhf, hi, Dhi, ho, Dho, hc, Dhc, Dc, y, pred])

    return caches, states



def LSTM_Sample(hprev, cprev):

    RandomInput = np.zeros([1, len(chars)])
    RandInteg = np.random.randint(0, len(chars))
    RandomInput[:, RandInteg] = 1.0
    nIteration = 1000
    exam = []
    for t in range(nIteration):

        x = np.column_stack([RandomInput, hprev])

        shf, sDhf = spike(np.dot(x, wf) + bf)
        shi, sDhi = spike(np.dot(x, wi) + bi)
        sho, sDho = spike(np.dot(x, wo) + bo)
        shc, sDhc = spike(np.dot(x, wc) + bc)
        #print(hf * c_prev + hi * hc)
        sc, sDc = spike(shf * cprev + shi * shc)
        sh = sho * sc

        hprev, cprev = sh, sc
        sy = np.dot(sh, wy) + by
        spred = softmax(sy).argmax()
        RandomInput = 0*RandomInput
        RandomInput[:, spred] = 1.0
        exam.append(spred)

    return exam



cnt = 0
Size_Train = xtrain.shape[0]

A = []
for i in range(ITER_NUM + 1):

    if cnt + BATCH_NUM >= Size_Train-1:
        cnt = 0

    X, Y = xtrain[cnt:cnt + BATCH_NUM], xtrain[cnt+1:cnt+ 1 + BATCH_NUM]

    cnt +=5

    caches, states  = LSTM_Cell(X)
    c, h = states[-1]

    dc_next = np.zeros_like(c)
    dh_next = np.zeros_like(h)
    CrossEntropy = 0

    if i%TextLOG == 0:
        GeneratedIndex= LSTM_Sample(h, c)
        txt = ''.join(indices_char[idx] for idx in GeneratedIndex)
        print(txt)

    for t in range(Time):

        x, hf, Dhf, hi, Dhi, ho, Dho, hc, Dhc, Dc, y, pred = caches[-t - 1]
        tc = c
        c, h = states[-t - 1]
        dout = pred - Y[-t - 1]
        CrossEntropy += cross_entropy(pred, Y[-t-1])
        c_prev, h_prev = states[-t - 2]

        dwy += np.dot(h.T, dout)
        dby += np.sum(dout, axis=0)

        dh = np.dot(dout, wy.T) + dh_next
        dc = dh * ho
        dc = 1* dc + dc_next

        dho = dh * tc
        dho = dho * deriv_spike(Dho)

        dhf = dc * deriv_spike(Dc) * c_prev
        dhf = dhf * deriv_spike(Dhf)

        dhi = dc * deriv_spike(Dc) * hc
        dhi = dhi * deriv_spike(Dhi)

        dhc = dc * deriv_spike(Dc) * hi
        dhc = dhc * deriv_Tanhspike(Dhc)

        dwy += np.dot(h.T, dout)
        dby += np.sum(dout, axis=0)

        dwf += np.dot(x.T, dhf)
        dbf += np.sum(dhf, axis=0)
        dXf = np.dot(dhf, wf.T)

        dwi += np.dot(x.T, dhi)
        dbi += np.sum(dhi, axis=0)
        dXi = np.dot(dhi, wi.T)

        dwo += np.dot(x.T, dho)
        # print(dwo)
        dbo += np.sum(dho, axis=0)
        dXo = np.dot(dho, wo.T)

        dwc += np.dot(x.T, dhc)
        dbc += np.sum(dhc, axis=0)
        dXc = np.dot(dhc, wc.T)

        dX = dXf + dXi + dXo + dXc

        dc_next = hf * dc
        dh_next = dX[:, -HIDDEN:]

    # Update weights

    mwf = (Beta1 * mwf + (1 - Beta1) * dwf)
    vwf = (Beta2 * vwf + (1 - Beta2) * (dwf ** 2))
    mwf_h = mwf / (1 - Beta1 ** (i + 1))
    vwf_h = vwf / (1 - Beta2 ** (i + 1))

    mwi = (Beta1 * mwi + (1 - Beta1) * dwi)
    vwi = (Beta2 * vwi + (1 - Beta2) * (dwi ** 2))
    mwi_h = mwi / (1 - Beta1 ** (i + 1))
    vwi_h = vwi / (1 - Beta2 ** (i + 1))

    mwc = (Beta1 * mwc + (1 - Beta1) * dwc)
    vwc = (Beta2 * vwc + (1 - Beta2) * (dwc ** 2))
    mwc_h = mwc / (1 - Beta1 ** (i + 1))
    vwc_h = vwc / (1 - Beta2 ** (i + 1))

    mwo = (Beta1 * mwo + (1 - Beta1) * dwo)
    vwo = (Beta2 * vwo + (1 - Beta2) * (dwo ** 2))
    mwo_h = mwo / (1 - Beta1 ** (i + 1))
    vwo_h = vwo / (1 - Beta2 ** (i + 1))

    mwy = (Beta1 * mwy + (1 - Beta1) * dwy)
    vwy = (Beta2 * vwy + (1 - Beta2) * (dwy ** 2))
    mwy_h = mwy / (1 - Beta1 ** (i + 1))
    vwy_h = vwy / (1 - Beta2 ** (i + 1))

    mbf = (Beta1 * mbf + (1 - Beta1) * dbf)
    vbf = (Beta2 * vbf + (1 - Beta2) * (dbf ** 2))
    mbf_h = mbf / (1 - Beta1 ** (i + 1))
    vbf_h = vbf / (1 - Beta2 ** (i + 1))

    mbi = (Beta1 * mbi + (1 - Beta1) * dbi)
    vbi = (Beta2 * vbi + (1 - Beta2) * (dbi ** 2))
    mbi_h = mbi / (1 - Beta1 ** (i + 1))
    vbi_h = vbi / (1 - Beta2 ** (i + 1))

    mbc = (Beta1 * mbc + (1 - Beta1) * dbc)
    vbc = (Beta2 * vbc + (1 - Beta2) * (dbc ** 2))
    mbc_h = mbc / (1 - Beta1 ** (i + 1))
    vbc_h = vbc / (1 - Beta2 ** (i + 1))

    mbo = (Beta1 * mbo + (1 - Beta1) * dbo)
    vbo = (Beta2 * vbo + (1 - Beta2) * (dbo ** 2))
    mbo_h = mbo / (1 - Beta1 ** (i + 1))
    vbo_h = vbo / (1 - Beta2 ** (i + 1))

    mby = (Beta1 * mby + (1 - Beta1) * dby)
    vby = (Beta2 * vby + (1 - Beta2) * (dby ** 2))
    mby_h = mby / (1 - Beta1 ** (i + 1))
    vby_h = vby / (1 - Beta2 ** (i + 1))

    # Update weights
    wf -= ALPHA * (mwf_h / (np.sqrt(vwf_h) + Epsilon))
    wi -= ALPHA * (mwi_h / (np.sqrt(vwi_h) + Epsilon))
    wc -= ALPHA * (mwc_h / (np.sqrt(vwc_h) + Epsilon))
    wo -= ALPHA * (mwo_h / (np.sqrt(vwo_h) + Epsilon))
    wy -= ALPHA * (mwy_h / (np.sqrt(vwy_h) + Epsilon))

    bf -= ALPHA * (mbf_h / (np.sqrt(vbf_h) + Epsilon))
    bi -= ALPHA * (mbi_h / (np.sqrt(vbi_h) + Epsilon))
    bc -= ALPHA * (mbc_h / (np.sqrt(vbc_h) + Epsilon))
    bo -= ALPHA * (mbo_h / (np.sqrt(vbo_h) + Epsilon))
    by -= ALPHA * (mby_h / (np.sqrt(vby_h) + Epsilon))

    # Initialize delta values
    dwf *= 0
    dwi *= 0
    dwc *= 0
    dwo *= 0
    dwy *= 0

    dbf *= 0
    dbi *= 0
    dbc *= 0
    dbo *= 0
    dby *= 0
    print('Iteration: {}'.format(i), 'Loss: {}'.format(CrossEntropy.mean()), 'Text Long: {}'.format(len(text)), 'Batch: {}'.format(cnt) )
    A.append(CrossEntropy.mean())




print(np.asarray(A).shape)
np.savetxt('entropyloss.txt', np.asarray(A), fmt='%8f')
