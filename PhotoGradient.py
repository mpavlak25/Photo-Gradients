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

"""
Planning/considering the problem:

At its very core, the problem of forming images into a gradient is a rather
extreme version of the traveling salesman problem. The challenging aspect comes
into place as  depending on the placement of tiles there are varying success
factors: it can be thought of as the ordering of tiles several tiles away affecting
the distance from one node to the next and in an uneven manner. Further, depending
on spatial positioning, nodes a and b may have a different distance.

Generating large datasets to generate adequate hyperparameters to act as
a fitness function for a genetic algorithm is prohibitive even if pulling a large
number of gradients off of social media. The function is important because the
final gradients will only be as good as the fitness function chosen. Basic
comparison operations will be made by measuring deltaE after converting to the
CAM02-UCS colorspace. These will be conducted on resized/shrunk images. The
measurements will need to take into account the spacial proximity of the images.
Further, images should be sampled throughout all corners throughout a gradient.
Genetic algorithms offer an interesting solution to the problem in the sense
that the mutation function can be tweaked to focus on mutating locally significant
areas: mutations a few photos away are much more likely to positively affect fitness
than one dozens away.
"""

"""approach two: basic genetic algorithms with a naive fitness function"""
def hashable(arr):
    arr2 = [map(tuple,subarr) for subarr in arr]
    return tuple(map(tuple,arr2))

class distanceDynamicPhoto():
    def __init__(self,image,images=[]):
        self.image = image
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



def openResize(pathTo,count,total):
    if (count%10==0):
        print(str(100*float(count)/total)+"%")
    im = cv2.cvtColor(cv2.imread(pathTo),cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(50,50))
    return im

def testValid(length,idx,cols):
    return (idx > 0) and (idx < length*cols)


def testFitness(filled,cols=3):
    #need to work
    score = 0
    filledFlat = [i for row in filled for i in row]
    for idx,im in enumerate(filledFlat):
        if testValid(len(filled),idx - 3,cols):
            #total number will be divided by three col will be it mod three
            # print(math.floor((idx-3)/3.0))
            # print(idx%3)
            score += im.distances[hash(filled[int(math.floor((idx-3)/3.0))][(idx-3)%3])][1]
        if testValid(len(filled),idx + 3,cols):
            print("filled")
            print(filled)
            print("dist")
            print((idx+3)%3)
            print(int(math.floor((idx+3)/3.0)),len(filled))


            print(filled[int(math.floor((idx+3)/3.0))])

            print(im.distances[hash(filled[int(math.floor((idx+3)/3.0))][(idx+3)%3])])
            print(math.floor((idx+3)/3.0))
            print(idx%3)
            score += im.distances[hash(filled[int(math.floor((idx+3)/3.0))][(idx+3)%3])][1]
        if testValid(len(filled),idx-1,cols):
            # print(math.floor((idx-1)/3.0))
            # print((idx-1)%3)
            score += im.distances[hash(filled[int(math.floor((idx-1)/3.0))][(idx-1)%3])][1]
        if testValid(len(filled),idx+1,cols):
            # print(math.floor((idx+1)/3.0))
            # print((idx+1)%3)
            score += im.distances[hash(filled[int(math.floor((idx+1)/3.0))][(idx+1)%3])][1]

    return score

path = "./Images/"
files = subprocess.check_output(["ls",path]).split("\n")
files = [i for i in files if i[-4:len(i)] == ".jpg"]

cs_converterTo1 = cs.cspace_converter("sRGB255","sRGB1")
cs_converterTo255 = cs.cspace_converter("sRGB1","sRGB255")

openImages = [cs_converterTo1(openResize(path+i,indx,len(files))) for indx,i in enumerate(files)]


# ###

# dynamicImages = [distanceDynamicPhoto(i,images=(openImages[0:indx]+openImages[indx+1:len(openImages)])) for indx,i in enumerate(openImages)]
#each has distances where hash of image and
dynamicImagesDict = {hash(hashable(i)):distanceDynamicPhoto(i,images=(openImages[0:indx]+openImages[indx+1:len(openImages)])) for indx,i in enumerate(openImages)}

#
origUnused = dynamicImagesDict.keys()

# print(origUnused)
fillSaves = []

seeds = 10
iterations = 20
for seed in range(0,seeds):

    toFill = [[-1,-1,-1] for i in range(0,int(math.ceil(len(dynamicImagesDict)/3.0)))]
    unused = list(origUnused)
    start = random.randint(0,len(dynamicImagesDict)-1)

    curr = dynamicImagesDict.keys()[start]
    # print(curr)
    # print("unused Length",str(len(unused)))
    # print(len(origUnused))
    for idx,row in enumerate(toFill):
        # print(toFill)
        # print("row:")
        # print(row)

        for i in range(0,len(row)):

            if len(unused) > 1:
                # print(curr)
                toFill[idx][i] = dynamicImagesDict[curr]
                # print("idx" +str(idx))
                #
                # print("hash to remove" + str(hash(hashable(dynamicImagesDict[curr].findNearest(unused)[0]))))
                # print(unused)
                # print(unused.index(curr),unused.remove(curr))
                print(unused,curr,len(unused))

                unused.remove(curr)
                curr = hash(hashable(dynamicImagesDict[curr].findNearest(list(unused))[0]))
            elif len(unused) == 1:
                toFill[idx][i] = dynamicImagesDict[curr]
                unused.remove(curr)

    # print(toFill)
    fillSaves.append(toFill)
print(fillSaves)

fitRes = map(testFitness,fillSaves)
print(map(testFitness,fillSaves))

#
#

####







#
# processedImages = openImages
#
#
#
# # distancePairs = [distancePhoto(i,np.sum(cs.deltaE(first,i))) for i in remaining]
# # sortedImages = [i.image for i in sorted(distancePairs, key=operator.attrgetter("distance"))]
# sortedImages = [processedImages[0]]
# closest = processedImages[0]
# remaining = openImages[1:len(openImages)]
#
# while (len(processedImages) > 1):
#     closest,processedImages = findNearest(processedImages[0],processedImages[1:len(processedImages)])
#     sortedImages.append(closest)
#
#
# toPlot = sortedImages
#
# for indx,images in enumerate(toPlot):
#     plt.subplot(math.ceil(len(toPlot)/7.0),7,indx+1)
#     plt.imshow(images)
#     plt.axis('off')
# plt.subplots_adjust(wspace=0,hspace=0)
# plt.show()






# print(np.sum(cs.deltaE(ArchesConv,GondolaConv)))
# print(np.sum(cs.deltaE(ArchesConv,KPSConv)))

# ArchesHSV = cv2.split(Arches)
# GondolaHSV = cv2.split(Gondola)
# KPSHSV = cv2.split(KPS)


#test naive implementation






#
# plt.subplot(1,3,1)
# plt.imshow(Arches)
# plt.subplot(1,3,2)
# plt.imshow(Gondola)
# plt.subplot(1,3,3)
# plt.imshow(KPS)

# plt.show()
