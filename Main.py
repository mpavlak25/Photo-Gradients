import subprocess, cv2, numpy as np, matplotlib.pyplot as plt, operator, math
import matplotlib.pyplot as plt, random, time, colorspacious as cs, argparse

from matplotlib import cm, colors

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

import multiprocessing as mp
import threading
import itertools
import base64

from tqdm import tqdm
import PhotoFunctions as pf
import DynamicPhoto
import Search
import cPickle,os

# import PhotoGradient



print("---------------------")
print("Photo Sort/Search App; Mitchell Pavlak")
print("---------------------")


# -*- coding: utf-8 -*-

# Print iterations progress"/Saves/savefile.pkl"
##based on https://stackoverflow.com/questions/2507808/how-to-check-whether-a-file-is-empty-or-not
def checkSave():
    if os.path.isfile("./Saves/savefile.pkl") and os.path.getsize("./Saves/savefile.pkl") > 0:
        return raw_input("Use found save? (y/n): ") == "y"

def loadSave():
    with open("./Saves/savefile.pkl") as f:
        order = cPickle.load(f)
    return order

def save(order,filename):
    cPickle.dumps(order,open(filename,"wb"))

def pairHelper(h):
    i = hashes.index(h)
    ret = []
    for curr in hashes[i+1:len(hashes)]:
        ret.append((h,curr))
    return ret

def calcDistWrapper(hashPair):
    return pf.calcDist(hashPair,dynamicImagesDict)

ncores = mp.cpu_count()

inpath = "./Images/"
sortedPath = "./Gradient"


if checkSave():
    order = loadSave()
    map(pf.convertStore,order)
    trackedFiles = [o.filename for o in order]

    files = subprocess.check_output(["ls",inpath]).split("\n")
    files = [i for i in files if i[-4:len(i)] == ".jpg"]
    fullPathFiles = [inpath+i for i in files]

    untracked = []
    removed = []

    for i in fullPathFiles:
        if i not in trackedFiles:
            untracked.append(i)
    for i in trackedFiles:
        if i not in fullPathFiles:
            removed.append(i)

    if (len(removed) != 0):
        print("Save files removed. Exiting...")
        exit(1)

    printer = [pf.openResize(i.filename) for i in order]

else:
    files = subprocess.check_output(["ls",inpath]).split("\n")
    files = [i for i in files if i[-4:len(i)] == ".jpg"]

    cs_converterToCIE = cs.cspace_converter("sRGB255","CAM02-UCS")
    cs_converterTo255 = cs.cspace_converter("CAM02-UCS","sRGB255")
    if __name__ == '__main__':
        openImages = []

        procPool = mp.Pool(ncores)
        filePath = [inpath+i for i in files]
        print("\n")
        print("---------------------")
        print("Found Files: ")
        print(files)
        print("---------------------")
        print("\n")

        print("Opening and resizing images...")
        for pathTo,im in tqdm(itertools.imap(pf.openResize,filePath),total = len(filePath)):
            openImages.append((pathTo,im))
        #opening appears proper.
        print("Opening / resize Completed.")
        print("")
        print("Converting to a perceptually uniform colorspace...")
        dynImages = []
        dynamicImagesDict = {}

        #(pathto, im)
        for i in tqdm(openImages,total=len(openImages)):
            dPhoto = DynamicPhoto.dynamicPhoto(cs_converterToCIE(i[1]),i[1],i[0])
            dynImages.append(dPhoto)

        print("Conversion complete.")
        print("")
        print("Quantizing colors...")
        dynImages = list(tqdm(procPool.imap(pf.quantWrapper,dynImages),total=len(dynImages)))
        print("Color quantization complete.")
        print("")
        print("Generating image dict...")
        for i in tqdm(dynImages):
            dynamicImagesDict[hash(i)] = i
        print("Generated.")
        print("")


        print("Generating hashlist and pairwise hashlist...")
        hashes = [hash(i) for i in dynImages]
        hashes = sorted(hashes)
        #####
        # def makePairwise(im): 1 2 3    12   13  23
        #     i = dynImages.index(im)  1 2 3 4 12 13 14 23 24 34
        #     while i < len(dynImages): 1 2 3 4 5 12 13 14 15 23 24 25 34 35 45
        #sum of 1 2 3 to n-1

        procPool.close()
        procPool.join()

        hashPool = mp.Pool(ncores)

        expectedLen = (len(hashes)*(len(hashes)-1))/2
        hashPairs = list(tqdm(hashPool.imap(pairHelper,hashes),total=expectedLen))
        hashPairs = [i for h in hashPairs for i in h]
        # print(hashPairs)
        # hashPairs = list(pf.makePairwise(hashes))
        print("Generation complete.")
        print("")

        print("Generating deltaE dictionary...")
        distDict = {}


            # print(val.colors)
        toAdd = list(tqdm(hashPool.imap(calcDistWrapper,hashPairs),total=len(hashPairs)))
        hashPool.close()
        hashPool.join()
        print("disc")
        # print(toAdd)
        print("dfcs")
        for i in toAdd:
            distDict[i[0]]=i[1]
        # for h in tqdm(hashPairs,total=len(hashPairs)):
        #     distDict[h]=pf.calcDist(h,dynamicImagesDict)
        print()
        print("deltaE dictionary generated.")
        print("Generating index dictionary...")
        indexDict = {h:indx for indx,h in enumerate(hashes)}
        print("Dictionary index generated.")

        print("Creating dataframe...")
        hashOrder,dF = pf.createDataFrame(indexDict,distDict)
        # print(dF)
        print("Dataframe generated.")
        print("")
        print("Optimizing...")
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
        search_param.time_limit.seconds = 20
        timerThread = threading.Thread(target=pf.timer,args=(20,))
        timerThread.start()
        ret = rt.SolveWithParameters(search_param)
        timerThread.join()
        print("Optimization complete.")



        toPlot = []
        if ret:
            ind = rt.Start(0)
            while not rt.IsEnd(ind):
                if ind != 0:
                    toPlot.append(dynamicImagesDict[hashOrder[ind-1]])
                ind = ret.Value(rt.NextVar(ind))
        printer = [i.rgbImage for i in toPlot]
        order = [DynamicPhoto.photoStore(i) for i in toPlot]
# print([order[0].colors])

toSave = raw_input("Would you like to save? y/n: ")
acceptable = ["yes","Yes","y","Y","YES"]
if toSave in acceptable:
    with open("./Saves/savefile.pkl",'wb') as saveFile:
        cPickle.dump(order,saveFile,0)


s = Search.Searcher()
time.sleep(1)
s.upload(order)
g = printer
toCheck = raw_input("enter s for search, g for gradient anything else to exit: ")
while toCheck in ["s","g"]:
    if toCheck == "s":
        toSearch = raw_input("Add search term: ")
        resultD = s.sSearch("&search="+toSearch)
        v = resultD["value".encode("utf-8")]
        print("Max to display is " +str(len(v)))
        num = int(raw_input("How many to display? "))


        ret = []
        for i in range(0,num):
            print(base64.urlsafe_b64decode(v[i]["filename".encode("utf-8")].encode("ASCII")))
            ret.append(inpath+base64.urlsafe_b64decode(v[i]["filename".encode("utf-8")].encode("ASCII"))+str(".jpg"))
        printer = []
        for name in ret:
            printer.append(pf.openResize(name))


        for i in printer:
            print(i[0])
        for indx,images in enumerate([i[1] for i in printer]):
            plt.subplot(math.ceil(len(printer)/3.0),3,indx+1)
            plt.imshow(images)
            plt.axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.show()
    elif toCheck == "g":
        for i in g:
            print(i[0])
        for indx,images in enumerate([i[1] for i in g]):
            plt.subplot(math.ceil(len(g)/3.0),3,indx+1)
            plt.imshow(images)
            plt.axis('off')
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.show()
    toCheck = raw_input("enter s for search, g for gradient anything else to exit: ")
