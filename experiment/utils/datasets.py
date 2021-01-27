#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import OrderedDict
import time
from experiment.utils.pre_data import read_xlsx, split_dataset

import numpy as np

f = "datasets/"


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def loadCodec():
    df = read_xlsx(f + 'codec/codec.xlsx')
    return df


def loadCollections():
    df = read_xlsx(f + 'collections/collections.xlsx')
    return df


def loadIo():
    df = read_xlsx(f + 'io/io.xlsx')
    return df


def loadJsoup():
    df = read_xlsx(f + 'jsoup/jsoup.xlsx')
    return df


def loadJsqlparser():
    df = read_xlsx(f + 'jsqlparser/jsqlparser.xlsx')
    return df


def loadMango():
    df = read_xlsx(f + 'mango/mango.xlsx')
    return df


def loadOrmlite():
    df = read_xlsx(f + 'ormlite/ormlite.xlsx')
    return df


d = OrderedDict()
s = time.time()

d["codec"] = loadCodec()
d["collections"] = loadCollections()
d["io"] = loadIo()
d["jsoup"] = loadJsoup()
d["jsqlparser"] = loadJsqlparser()
d["mango"] = loadMango()
d["ormlite"] = loadOrmlite()

print("Data loaded in {:5.2f}".format(time.time()-s))


