from __future__ import print_function

import time
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd
import os

import numpy as np

CLASS_NUM=10

train = pd.read_csv("emnist-mnist-train.csv")
test = pd.read_csv("emnist-mnist-test.csv")

train = np.asarray(train).astype('float32')
test = np.asarray(test).astype('float32')

xtrain = train[:, 1:]
xtest = test[:, 1:]

xtrain /= 255
xtest /= 255

ytrain = train[:, 0]
ytest = test[:, 0]
CLASS_NUM = 10


ytrain = np_utils.to_categorical(ytrain, CLASS_NUM)
ytest = np_utils.to_categorical(ytest, CLASS_NUM)

INPUT = 28
HIDDEN = 200
OUTPUT = 10
Time = 28

INPUT += HIDDEN

ALPHA = 0.001
Beta1 = 0.9
Beta2 = 0.999
Epsilon = 1e-8

BATCH_NUM = 100

ITER_NUM = 200
theta = 0.1


LOG = 1
Size_Train = ytrain.shape[0]
Size_Test = ytest.shape[0]
errors = [] # to plot learning curve of cross entropy

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

A = []
def softmax(arr):
    c = np.clip(arr, -700, 700) # float64 maximum expotentiable value
    e = np.exp(c)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(out, label):
    entropy = label * np.log(out + 1e-6) # to prevent log value overflow
    return -np.sum(entropy, axis=1, keepdims=True)


def spike(arr):
    C = 1 * (arr >theta)
    Dif = arr - theta
    return C, Dif

def deriv_spike(out):
    return (1/np.sqrt(0.3*np.pi)) * np.exp(-out**2/0.3)

def deriv_Tanhspike(out):
    return (1/np.sqrt(0.3*np.pi)) * np.exp(-out**2)

def Test():
    images, labels = xtest, ytest
    _, states = LSTM_Cell(images)
    _, h = states[-1]
    out = softmax(np.dot(h, wy) + by)
    entropy = np.mean(cross_entropy(out, labels))
    pred = np.argmax(out, axis=1)
    labels = np.argmax(labels, axis=1)
    Acc = np.mean(pred == labels)

    return Acc, entropy


def predict(img):
    input_val = img

    caches, states = LSTM_Cell(input_val)
    c, h = states[-1]

    pred = softmax(np.dot(h, wy) + by)
    label = np.argmax(pred)

    return label

mwf, mwi, mwc, mwo, mwy = 0, 0, 0, 0, 0
vwf, vwi, vwc, vwo, vwy = 0, 0, 0, 0, 0


mbf, mbi, mbc, mbo, mby = 0, 0, 0, 0, 0
vbf, vbi, vbc, vbo, vby = 0, 0, 0, 0, 0


def LSTM_Cell(input_val):
    batch_num = input_val.shape[0]
    caches = []
    states = []
    states.append([np.zeros([batch_num, HIDDEN]), np.zeros([batch_num, HIDDEN])])
    p = np.random.uniform(0, 1, np.shape(input_val))
    input_val = 1*(input_val>p)

    for t in range(Time):

        splitedInput = input_val[:, t*Time:(t+1)*Time]
        c_prev, h_prev = states[-1]

        x = np.column_stack([splitedInput, h_prev])

        hf, Dhf = spike(np.dot(x, wf) + bf)
        hi, Dhi = spike(np.dot(x, wi) + bi)
        ho, Dho = spike(np.dot(x, wo) + bo)
        hc, Dhc = spike(np.dot(x, wc) + bc)
        #print(hf * c_prev + hi * hc)
        c, Dc = spike(hf * c_prev + hi * hc)

        h = ho * c

        states.append([c, h])
        caches.append([x, hf, Dhf, hi, Dhi, ho, Dho, hc, Dhc, Dc])

    return caches, states


cnt = 0
for i in range(ITER_NUM + 1):

    if cnt + BATCH_NUM >= Size_Train:
        cnt = 0

    X, Y = xtrain[cnt:cnt + BATCH_NUM, :], ytrain[cnt:cnt + BATCH_NUM, :]
    cnt = cnt + BATCH_NUM


    caches, states = LSTM_Cell(X)
    c, h = states[-1]
    out = np.dot(h, wy) + by
    pred = softmax(out)
    entropy = cross_entropy(pred, Y)
    dout = pred - Y

    dwy = np.dot(h.T, dout)
    dby = np.sum(dout, axis=0)

    dc_next = np.zeros_like(c)
    dh_next = np.zeros_like(h)

    for t in range(Time):
        c, h = states[-t - 1]
        c_prev, h_prev = states[-t - 2]

        x, hf, Dhf, hi, Dhi, ho, Dho, hc, Dhc, Dc = caches[-t - 1]
        tc = c

        dh = np.dot(dout, wy.T) + dh_next

        dc = dh * ho
        dc = 1 * dc + dc_next

        dho = dh * tc
        dho = dho * deriv_spike(Dho)

        dhf = dc * deriv_Tanhspike(Dc) * c_prev
        dhf = dhf * deriv_spike(Dhf)

        dhi = dc * deriv_Tanhspike(Dc) * hc
        dhi = dhi * deriv_spike(Dhi)

        dhc = dc * deriv_Tanhspike(Dc) * hi
        dhc = dhc * deriv_Tanhspike(Dhc)

        dwf += np.dot(x.T, dhf)
        dbf += np.sum(dhf, axis=0)
        dXf = np.dot(dhf, wf.T)

        dwi += np.dot(x.T, dhi)
        dbi += np.sum(dhi, axis=0)
        dXi = np.dot(dhi, wi.T)

        dwo += np.dot(x.T, dho)
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

    A.append(entropy.mean())
    if i % LOG == 0:
        predicted = np.argmax(out, axis=1)
        Tlabels = np.argmax(Y, axis=1)
        TAcc = np.mean(predicted == Tlabels)

        print(TAcc, entropy.mean())
        print(Test())

np.savetxt('MNSIT.txt', np.asarray(A), fmt='%8f')
