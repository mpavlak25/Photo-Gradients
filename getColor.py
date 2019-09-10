import subprocess, cv2, numpy as np, matplotlib.pyplot as plt, operator, math
import matplotlib.pyplot as plt, random, time, colorspacious as cs, argparse
from matplotlib import cm, colors
import itertools

from tqdm import tqdm
import PhotoFunctions as pf
import DynamicPhoto
import extractMetadata
import cPickle,os
import heapq
############################
import json,datetime
import requests
from pprint import pprint
class getColor(object):
    def concatColors(self,l):
        a = ""
        for i in l:
            a += i + " "
        return a
    def __init__(self):
        cs_converterToCIE = cs.cspace_converter("sRGB255","CAM02-UCS")
        d = {}
        with open("color.txt",'r') as f:
            for line in f:
                if line[0] != "!":
                    l = line.split()

                    d[tuple(np.ndarray.tolist(cs_converterToCIE((int(l[0]),int(l[1]),int(l[2])))))] = self.concatColors(l[3:len(l)])
        self.colorDict = d
    def closest2(self,color):
        l = heapq.nsmallest(2,self.colorDict.items(),key = lambda k:pf.cdist(k[0],color))
        return l
