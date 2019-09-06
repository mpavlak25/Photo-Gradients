import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
import colorspacious as cs

import operator
import math
import random
import time

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from sklearn.cluster import MiniBatchKMeans

from collections import Counter

"""
Ok so good on the general implementation. What I do need to do is make the ms
azure piece/finish it

goals is convert to cie first
cluster first and all references
either cluster separately or together and keep track of indices



Ok so having trouble implementing, gonna try pytorch or keras and see if I can
get it done

to do
switch to kmeans color quanitzation and scrap whole shebang of stuff so it's more efficient
use pytorch or keras and get azure ml worked into my solution


faster photo import


Planning/considering the problem:

At its very core, the problem of forming images into a gradient is a rather
extreme version of the traveling salesman problem. In an optimal solution,
factors such as relative positioning matter.
After attempting a naive and basic solution, I attempted to solve using a
genetic algorithm; however, I found that when considering relative position as
well as those above and below, I rather quickly ran into problems where there
were a significant number of global minima, and to increase randomness and
iterations to the point that I would move beyond those minima seemed to take a
substantial amount of processing time for a realistic (and large) number of
photos.

My next goal is to use Microsoft Azure to improve direct image comparisons and
solely use non spatially dependent one to one comparisons to narrow down the
solution space.

Basic comparison operations will be made by measuring deltaE after converting
to the CAM02-UCS colorspace. These will be conducted on resized/shrunk images. The
measurements will need to take into account the spacial proximity of the images.
Further, images should be sampled throughout all corners throughout a gradient.
Genetic algorithms offer an interesting solution to the problem in the sense
that the mutation function can be tweaked to focus on mutating locally significant
areas: mutations a few photos away are much more likely to positively affect fitness
than one dozens away.
"""

"""approach two: basic genetic algorithms with a naive fitness function"""


def openResize(pathTo,count,total,res = 500):
    if (count%10==0):
        print(str(100*float(count)/total)+"%")
    im = cv2.cvtColor(cv2.imread(pathTo),cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(res,res))
    return im

def hashable(arr):
    """Returns a hashable form of an image where all the lists are replaced with
    tuples"""
    arr2 = [map(tuple,subarr) for subarr in arr]
    return tuple(map(tuple,arr2))

class dynamicPhoto():
    def __init__(self,image,rgbImage):
        self.rgbImage = rgbImage
        self.image = image
        self.total = 0
        self.colors = {}
    def colorQuant(self,centroids):
        mbKmeans = MiniBatchKMeans(n_clusters=centroids)
        # print(type(self.image))
        mbkFitted = mbKmeans.fit(self.image.reshape(self.image.shape[0]*self.image.shape[1],3))
        colors = mbkFitted.cluster_centers_
        count = Counter(mbkFitted.labels_)
        # print(count)
        self.total = float(len(mbkFitted.labels_))
        # print(self.total)
        colors = map(tuple,colors)

        self.colors = {i:count[indx]/self.total for indx,i in enumerate(colors)}

    def __hash__(self):
        return hash(hashable(self.rgbImage))

### organization
# -- first setup for importing all photos

#get names
path = "./Images/"
files = subprocess.check_output(["ls",path]).split("\n")
files = [i for i in files if i[-4:len(i)] == ".jpg"]
# get colorspace batch converters
cs_converterToCIE = cs.cspace_converter("sRGB255","CAM02-UCS")
cs_converterTo255 = cs.cspace_converter("CAM02-UCS","sRGB255")


# -- load them up in a hashable form
openImages = [openResize(path + i, indx, len(files)) for indx,i in enumerate(files)]
dynImages = [dynamicPhoto(cs_converterToCIE(i),i) for i in openImages]
dynamicImagesDict = {hash(i):i for i in dynImages}

# -- perform color quantization on the photos
for images in dynImages:
    images.colorQuant(16)

# list all the hash values of the photos
hashes = [hash(i) for i in dynImages]

# # make a dictionary of pairwise distances
def makePairwise(listA):
    for indx,items in enumerate(listA):
        for others in listA[indx+1:]:
            if others < items:
                yield others,items
            else:
                yield items,others

hashPairs = list(makePairwise(hashes))


#
# create a distance matrix
def cdist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))
def calcDist(hashPair,dynamicImagesDict):
    dist = 0
    print(dynamicImagesDict[hashPair[0]])
    for colors in dynamicImagesDict[hashPair[0]].colors.items():
        for otherColors in dynamicImagesDict[hashPair[1]].colors.items():
            # print(colors)
            # print(otherColors)
            dist += colors[1]*otherColors[1]*cdist(colors[0],otherColors[0])
    return dist

distDict = {h:calcDist(h,dynamicImagesDict) for h in hashPairs}

indexDict = {h:indx for indx,h in enumerate(hashes)}


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

hashOrder,dF = createDataFrame(indexDict,distDict)

manager = pywrapcp.RoutingIndexManager(len(dF["distance_matrix"]),
dF["num_vehicles"],dF["start"])

rt = pywrapcp.RoutingModel(manager)
def distance_call(fromI,toI):
    fNode = manager.IndexToNode(fromI)
    tNode = manager.IndexToNode(toI)
    return dF["distance_matrix"][fNode][tNode]
cbIndex = rt.RegisterTransitCallback(distance_call)
rt.SetArcCostEvaluatorOfAllVehicles(cbIndex)

search_param = pywrapcp.DefaultRoutingSearchParameters()
search_param.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_param.time_limit.seconds = 15
t = time.time()
ret = rt.SolveWithParameters(search_param)
print("--- %s seconds ---" % (time.time() - t))
toPlot = []
if ret:
    ind = rt.Start(0)
    while not rt.IsEnd(ind):
        if ind != 0:
            toPlot.append(dynamicImagesDict[hashOrder[ind-1]].rgbImage)
        ind = ret.Value(rt.NextVar(ind))

for indx,images in enumerate(toPlot):
    plt.subplot(math.ceil(len(toPlot)/3.0),3,indx+1)
    plt.imshow(images)
    plt.axis('off')
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()
