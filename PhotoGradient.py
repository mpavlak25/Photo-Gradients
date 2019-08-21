import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import colors
import colorspacious as cs
import heapq
import operator
import math
import random
import time

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

"""
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


def hashable(arr):
    """Returns a hashable form of an image where all the lists are replaced with
    tuples"""
    arr2 = [map(tuple,subarr) for subarr in arr]
    return tuple(map(tuple,arr2))

class distanceDynamicPhoto():
    def __init__(self,image,Himage,images=[]):
        self.image = image
        self.Himage = Himage
        self.distances = {}
        # print("")
        # print(np.shape(images))
        # print("")
        for i in images:
            self.distances[hash(hashable(i))] = (i,np.sum(cs.deltaE(self.image,i)))
        # print([i[1] for i in self.distances.values()])
    def addDistance(self,oImage):
        self.distances[hash(hashable(oImage))] = (oImage,np.sum(cs.deltaE(self.image,oImage)))
    def addDistances(self,oImages):
        for images in oImages:
            self.distances[hash(hashable(images))] = (image,np.sum(cs.deltaE(self.image,images)))
    def __hash__(self):
        return hash(hashable(self.image))
    def findNearest(self,NotUsedHash):

        dist = [self.distances[nu] for nu in NotUsedHash if nu != hash(self)]
        # for i in dist:
        #     print("shape" +str(np.shape(i)))
        #     if str(np.shape(i))=="()":
        #         print(i)
        # print(dist)
        # print(min(self.distances.values(),key=operator.itemgetter(1)))
        return min(dist,key=operator.itemgetter(1))

def openResize(pathTo,count,total,res = 8):
    if (count%10==0):
        print(str(100*float(count)/total)+"%")
    im = cv2.cvtColor(cv2.imread(pathTo),cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(res,res))
    return im



path = "./Images/"
files = subprocess.check_output(["ls",path]).split("\n")
files = [i for i in files if i[-4:len(i)] == ".jpg"]
print(files)
cs_converterTo1 = cs.cspace_converter("sRGB255","sRGB1")
cs_converterTo255 = cs.cspace_converter("sRGB1","sRGB255")

openImages = [(cs_converterTo1(openResize(path+i,indx,len(files))),openResize(path+i,1,1,res=100)) for indx,i in enumerate(files)]

print(openImages)

# ###

# dynamicImages = [distanceDynamicPhoto(i,images=(openImages[0:indx]+openImages[indx+1:len(openImages)])) for indx,i in enumerate(openImages)]
#each has distances where hash of image and
dynImageList = [distanceDynamicPhoto(i[0],i[1],images=([a[0] for a in openImages
if hash(hashable(a[0])) != hash(hashable(i[0]))])) for i in openImages]
dynamicImagesDict = {hash(i):i for i in dynImageList}




# dynamicImagesDict = {hash(hashable(i)):distanceDynamicPhoto(i[0],i[1],images=(
# list(zip(*openImages[0:indx])[0])+list(zip(*openImages[indx+1:len(
# openImages)])[0]))) for indx,i in enumerate(openImages)}


origUnused = dynamicImagesDict.keys()


print("num")
print(len(origUnused))





def createDataFrame(dynamicImagesDict,scale = 10000):
    hashList = dynamicImagesDict.keys()
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
                temp.append(int(dynamicImagesDict[dImagesH].distances[h][1]*scale))
        distanceMatrix.append(temp)
    dF = {"distance_matrix":distanceMatrix}
    dF["start"] = 0
    dF["num_vehicles"] = 1
    return hashList,dF

hashOrder,dF = createDataFrame(dynamicImagesDict)

# print(hashOrder)
# print(dF)
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
    # print(ret.ObjectiveValue())
    ind = rt.Start(0)
    while not rt.IsEnd(ind):
        # print("---")
        # print(manager.IndexToNode(ind))
        # print(ind)
        if ind != 0:
            # print(hashOrder[ind-1])
            toPlot.append(dynamicImagesDict[hashOrder[ind-1]].Himage)
        # print("---")
        ind = ret.Value(rt.NextVar(ind))



# # print(fittestSolutions)
# print("")
# # print(min(fittestSolutions,key=operator.itemgetter(0)))
# print(sorted(sanity,key=operator.itemgetter(0)))
# # fitRes = map(testFitness,fillSaves)
#
# toPrint = min(fittestSolutions,key=operator.itemgetter(0))[1]
# toPlot = [dynamicImagesDict[i].image for row in toPrint for i in row]
# print(toPlot)



for indx,images in enumerate(toPlot):
    plt.subplot(math.ceil(len(toPlot)/3.0),3,indx+1)
    plt.imshow(images)
    plt.axis('off')
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()

# print(origUnused)
fillSaves = []
#
# seeds = 40
# iterations = 20
# for seed in range(0,seeds):
#
#     toFill = [[-1,-1,-1] for i in range(0,int(math.ceil(len(dynamicImagesDict)/3.0)))]
#     unused = list(origUnused)
#     start = random.randint(0,len(dynamicImagesDict)-1)
#
#     curr = dynamicImagesDict.keys()[start]
#
#     for idx,row in enumerate(toFill):
#
#         for i in range(0,len(row)):
#
#             if len(unused) > 1:
#                 # print(curr)
#                 toFill[idx][i] = curr
#
#
#                 unused.remove(curr)
#                 curr = hash(hashable(dynamicImagesDict[curr].findNearest(list(unused))[0]))
#             elif len(unused) == 1:
#                 toFill[idx][i] = curr
#                 unused.remove(curr)
#
#     # print(toFill)
#     # for row in toFill:
#     #     random.shuffle(row)
#     fillSaves.append(toFill)
# # print(fillSaves)
#
#
# def testValid(length,idx,cols):
#     return (idx > 0) and (idx < length*cols)
#
#
# def testFitness(filled,cols=3):
#     #need to work
#     score = 0
#     filledFlat = [i for row in filled for i in row]
#     weightNext = 1.3
#     for idx,h in enumerate(filledFlat):
#         if testValid(len(filled),idx - 3,cols):
#
#             score += dynamicImagesDict[h].distances[filled[int(math.floor((idx-3)/3.0))][(idx-3)%3]][1]
#         if testValid(len(filled),idx + 3,cols):
#
#             # print(idx%3)
#             score += dynamicImagesDict[h].distances[filled[int(math.floor((idx+3)/3.0))][(idx+3)%3]][1]
#         if testValid(len(filled),idx-1,cols):
#
#             score += weightNext*dynamicImagesDict[h].distances[filled[int(math.floor((idx-1)/3.0))][(idx-1)%3]][1]
#         if testValid(len(filled),idx+1,cols):
#
#             score += weightNext*dynamicImagesDict[h].distances[filled[int(math.floor((idx+1)/3.0))][(idx+1)%3]][1]
#
#     return score,filled
#
#
# def selectFittest(fitRes,topAmount = .7,numRand = .3,origlen=40):
#     sortedFit = sorted(fitRes,key=operator.itemgetter(0))
#     Fit = sortedFit[0:int(origlen*topAmount)]
#     remaining = sortedFit[int(origlen*topAmount):len(sortedFit)]
#     numRandom = int(len(sortedFit)*numRand)
#     for i in range(0,numRandom):
#         Fit.append(remaining.pop(random.randint(0,len(remaining)-1)))
#     return Fit
#
# def mutate(fit,probOfRemove = .02,stopProb=.2):
#     options = [list(np.ndarray.flatten(np.array(i[1]))) for i in fit]
#     scores = np.array([i[0] for i in fit])
#     # scoresD = scores/scores.mean()
#     finalOptions = []
#     for indx,option in enumerate(options):
#         # print(option)
#         origOpt = np.ndarray.tolist(np.array(option).reshape(-1,3))
#         finalOptions.append(origOpt)
#
#         temp = list(option)
#         inRemove = False
#         toAdd = []
#         for i in range(0,len(temp)-1):
#             if inRemove:
#                 if random.random() < stopProb:
#                     inRemove = False
#                 else:
#
#                     option.remove(temp[i])
#                     toAdd.append(temp[i])
#             # elif random.random() < probOfRemove+scoresD[indx]:
#             elif random.random() < probOfRemove:
#                 option.remove(temp[i])
#                 toAdd.append(temp[i])
#                 inRemove = True
#         # print(option)
#         # print(toAdd)
#         # print("up is add")
#         for removed in toAdd:
#             nearest = hash(hashable(dynamicImagesDict[removed].findNearest(option)[0]))
#             option.insert(nearest,removed)
#         option = np.ndarray.tolist(np.array(option).reshape(-1,3))
#         finalOptions.append(option)
#         for i in range(0,random.randint(0,int(math.pow(len(temp),.5)/1.5))):
#             a = random.randint(0,len(temp)-1)
#             b = random.randint(0,len(temp)-1)
#             temp[a],temp[b] = temp[b],temp[a]
#         temp = np.ndarray.tolist(np.array(temp).reshape(-1,3))
#         finalOptions.append(temp)
#
#
#     return finalOptions
#
#
#
#
#
# fittestSolutions = []
# sanity = []
#
# for i in range(0,80):
#     # if (i%10 == 0):
#     print("Iteration:" +str(i))
#
#     fitScored = map(testFitness,fillSaves)
#     # print(len(fitScored))
#     generation = selectFittest(fitScored)
#     # print(len(generation))
#     fittestSolutions.append(min(fitScored,key=operator.itemgetter(0)))
#     sanity.append((fittestSolutions[-1][0],i))
#     fillSaves = mutate(generation)
#     # print(fillSaves)
#
#

# #
# #
# #
# # fitRes = selectFittest(fitRes)
# #
# # print(np.shape(fitRes))
# #
#
#
#
#
# # # print(options)
# # mapping = {hash(dphoto):dphoto for row in options[0] for dphoto in row}
# #
# #
#
#
#
# # bestNaiveIdx = fitRes.index(min(fitRes))
# # bestNaive = fillSaves[bestNaiveIdx]
# # #
# # # print(bestNaive)
# #
# #
# #
# #
# #
# # toPlot = [dI.image for row in bestNaive for dI in row]
# #
# # for indx,images in enumerate(toPlot):
# #     plt.subplot(math.ceil(len(toPlot)/3.0),3,indx+1)
# #     plt.imshow(images)
# #     plt.axis('off')
# # plt.subplots_adjust(wspace=0,hspace=0)
# # plt.show()
#
#
# #
# #
#
# ####
#
#
#
#
#
#
#
# #
# # processedImages = openImages
# #
# #
# #
# # # distancePairs = [distancePhoto(i,np.sum(cs.deltaE(first,i))) for i in remaining]
# # # sortedImages = [i.image for i in sorted(distancePairs, key=operator.attrgetter("distance"))]
# # sortedImages = [processedImages[0]]
# # closest = processedImages[0]
# # remaining = openImages[1:len(openImages)]
# #
# # while (len(processedImages) > 1):
# #     closest,processedImages = findNearest(processedImages[0],processedImages[1:len(processedImages)])
# #     sortedImages.append(closest)
# #
# #
#
#
#
#
#
#
#
# # print(np.sum(cs.deltaE(ArchesConv,GondolaConv)))
# # print(np.sum(cs.deltaE(ArchesConv,KPSConv)))
#
# # ArchesHSV = cv2.split(Arches)
# # GondolaHSV = cv2.split(Gondola)
# # KPSHSV = cv2.split(KPS)
#
#
# #test naive implementation
#
#
#
#
#
#
# #
# # plt.subplot(1,3,1)
# # plt.imshow(Arches)
# # plt.subplot(1,3,2)
# # plt.imshow(Gondola)
# # plt.subplot(1,3,3)
# # plt.imshow(KPS)
#
# # plt.show()
