import subprocess, cv2, numpy as np, matplotlib.pyplot as plt, operator, math
import matplotlib.pyplot as plt, random, time, colorspacious as cs, argparse

from matplotlib import cm, colors

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

import multiprocessing as mp
import itertools

from tqdm import tqdm
import PhotoFunctions as pf


class dynamicPhoto(object):
    def __init__(self,image,rgbImage,filename):
        self.rgbImage = rgbImage
        self.image = image
        self.filename = filename
        self.total = 0
        self.colors = {}
        self.lens = None
        self.date = None
        self.camera = None

    def colorQuant(self,centroids):
        mbKmeans = MiniBatchKMeans(n_clusters=centroids)
        mbkFitted = mbKmeans.fit(self.image.reshape(self.image.shape[0]*self.image.shape[1],3))
        colors = mbkFitted.cluster_centers_
        count = Counter(mbkFitted.labels_)
        self.total = float(len(mbkFitted.labels_))
        colors = map(tuple,colors)

        self.colors = {i:count[indx]/self.total for indx,i in enumerate(colors)}
        return self

    def __hash__(self):
        return hash(pf.hashable(self.rgbImage))

class photoStore(object):
    def __init__(self,photo):
        self.filename = photo.filename
        self.total = photo.total
        self.colors = photo.colors
        self.lens = None
        self.date = None
        self.camera = None
