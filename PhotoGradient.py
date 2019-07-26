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

class distancePhoto():
    def __init__(self,image,distance):
        self.image = image
        self.distance = distance
    def __eq__(self,o):
        return self.image == o.image and self.distance == o.distance
    def __lt__(self,o):
        return self.distance < o.distance
    def __le__(self,o):
        return self.distance <= o.distance
    def __gt__(self,o):
        return self.distance > o.distance
    def __ge__(self,o):
        return self.distance >= o.distance
    def __hash__(self):
        return hash(image,distance)

def openResize(pathTo,count,total):
    if (count%10==0):
        print(str(100*float(count)/total)+"%")
    im = cv2.cvtColor(cv2.imread(pathTo),cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(50,50))
    return im

def findNearest(first,rest):
    distancePairs = [distancePhoto(i,np.sum(cs.deltaE(first,i))) for i in rest]
    heapq.heapify(distancePairs)
    return heapq.heappop(distancePairs).image,[i.image for i in distancePairs]

path = "./Images/"
files = subprocess.check_output(["ls",path]).split("\n")
files = [i for i in files if i[-4:len(i)] == ".jpg"]


cs_converterTo1 = cs.cspace_converter("sRGB255","sRGB1")
cs_converterTo255 = cs.cspace_converter("sRGB1","sRGB255")

openImages = [cs_converterTo1(openResize(path+i,indx,len(files))) for indx,i in enumerate(files)]

processedImages = openImages



# distancePairs = [distancePhoto(i,np.sum(cs.deltaE(first,i))) for i in remaining]
# sortedImages = [i.image for i in sorted(distancePairs, key=operator.attrgetter("distance"))]
sortedImages = [processedImages[0]]
closest = processedImages[0]
remaining = openImages[1:len(openImages)]

while (len(processedImages) > 1):
    closest,processedImages = findNearest(processedImages[0],processedImages[1:len(processedImages)])
    sortedImages.append(closest)


toPlot = sortedImages

for indx,images in enumerate(toPlot):
    plt.subplot(math.ceil(len(toPlot)/7.0),7,indx+1)
    plt.imshow(images)
    plt.axis('off')
plt.subplots_adjust(wspace=0,hspace=0)
plt.show()






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

plt.show()
