import subprocess, cv2, numpy as np, matplotlib.pyplot as plt, operator, math
import matplotlib.pyplot as plt, random, time, colorspacious as cs, argparse

from matplotlib import cm, colors

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

import multiprocessing as mp
import itertools
import DynamicPhoto as dp

from tqdm import tqdm





def convertStore(stored):
    print(stored.filename)
    try:
        im = openResize(stored.filename)
        PercepUniIm = cs.cspace_convert(im[1],"sRGB255","CAM02-UCS")
        p = dp.dynamicPhoto(PercepUniIm,im,stored.filename)
        p.total = stored.total
        p.colors = stored.colors
        return p
    except cv2.error:
        print(stored.filename + " not found.")


def openResize(pathTo,res = 500):
    im = cv2.cvtColor(cv2.imread(pathTo),cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(res,res))
    return pathTo,im

def hashable(arr):
    """Returns a hashable form of an image where all the lists are replaced with
    tuples"""
    arr2 = [map(tuple,subarr) for subarr in arr]
    return tuple(map(tuple,arr2))



def cdist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))
def calcDist(hashPair,dynamicImagesDict):
    dist = 0
    for colors in dynamicImagesDict[hashPair[0]].colors.items():
        for otherColors in dynamicImagesDict[hashPair[1]].colors.items():
            dist += colors[1]*otherColors[1]*cdist(colors[0],otherColors[0])
    return (hashPair,dist)

def createDataFrame(indexDict,distDict,scale = 10000):
    hashList = indexDict.keys()
    distanceMatrix = []
    #want a matrix that has space for all in hashlist+1
    distanceMatrix.append([0 for i in range(0,len(hashList)+1)])
    #want to make a null node which will always have zero distance
    for currIdx,dImagesH in enumerate(hashList):
        temp = [0]
        for idx,h in enumerate(hashList):
            if currIdx == idx:
                temp.append(0)
            else:
                if dImagesH < h:
                    temp.append(int(distDict[(dImagesH,h)]*scale))
                elif h < dImagesH:
                    temp.append(int(distDict[(h,dImagesH)]*scale))
        distanceMatrix.append(temp)
    dF = {"distance_matrix":distanceMatrix}
    dF["start"] = 0
    dF["num_vehicles"] = 1
    return hashList,dF
def quantWrapper(im,centroids=16):
    return im.colorQuant(centroids)
def timer(sec):
    for i in tqdm(range(0,sec)):
        time.sleep(1)
