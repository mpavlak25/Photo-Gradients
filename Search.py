import subprocess, cv2, numpy as np, matplotlib.pyplot as plt, operator, math
import matplotlib.pyplot as plt, random, time, colorspacious as cs, argparse
from matplotlib import cm, colors
import itertools

from tqdm import tqdm
import PhotoFunctions as pf
import DynamicPhoto
import getColor
import extractMetadata
import cPickle,os
import heapq
import base64

############################
import json,datetime
import requests
from pprint import pprint



# https://photo-search.search.windows.net
# Copy to clipboard
# e11810c3-51f8-4724-8211-a8ddd2c22315
# 4D35AC98C91E3E31562294C3BF11E584
class Searcher(object):

    def __init__(self,file_source = "./Images"):
        self.file_source = file_source
        extractorObj = extractMetadata.extractor(file_source)
        self.metadataDict = extractorObj.extract_metadata()
        self.endpoint = "https://photo-search.search.windows.net/"
        self.api_version = '?api-version=2019-05-06'
        self.headers = {'Content-Type': 'application/json',
            'api-key': '4D35AC98C91E3E31562294C3BF11E584'}
        url = self.endpoint + "indexes" + self.api_version + "&$select=name"
        response  = requests.get(url, headers=self.headers)
        index_list = response.json()
        if len(index_list["value".encode("utf-8")]) == 0:
            self.formatIndex()

    def formatIndex(self):
        index_schema = {
            "name": "images",
            "fields": [
            {"name": "filename", "type": "Edm.String", "key": "true", "filterable": "false"},
            {"name": "Camera", "type": "Edm.String", "searchable": "true", "filterable": "true", "sortable": "false", "facetable": "true"},
            {"name": "Lens", "type": "Edm.String", "searchable": "true", "filterable": "true", "sortable": "false", "facetable": "true"},
            {"name": "Date", "type": "Edm.DateTimeOffset", "filterable": "true", "sortable": "true", "facetable": "true"},
            {"name": "colors", "type": "Edm.String", "searchable": "true", "filterable": "true", "sortable": "true", "facetable": "true"},

            ]
        }
        url = self.endpoint + "indexes" + self.api_version
        response  = requests.post(url, headers=self.headers, json=index_schema)
        try:
            print(response.json())
        except:
            print("An index creation error occured.")

    def genColors(self,im):
        retStr = ""
        l = im.colors.items()
        l = sorted(l,reverse=True,key=lambda i: i[1])
        scaled = [(i[0],int(i[1]*100)) for i in l]
        gC = getColor.getColor()
        for i in scaled:
            closest = gC.closest2(i[0])
            retStr += ((closest[0][1]+" ")*i[1])

        return retStr

    def updateMeta(self,i):
        iMeta = self.metadataDict[i.filename[len(self.file_source)+1:len(i.filename)]]
        try:
            i.camera = iMeta["Canon Model ID"]
        except:
            i.camera = "Not found"
            try:
                i.camera = iMeta["Camera Model Name"]
            except:
                pass
        try:
            i.lens = iMeta["Lens ID"]
        except:
            i.lens = "Not found"
        try:
            d = iMeta['Date/Time Original']
            d = str(datetime.datetime(int(d[0:4]),int(d[5:7]),int(d[11:13])))[0:10]+"T00:00:00Z"
            i.date = d
        except:
            i.date = str(datetime.datetime.min)[0:10]+"T00:00:00Z"
        print(i.camera,i.date,i.lens)



    def upload(self,imStorewMeta):
        v = []

        for i in imStorewMeta:
            c = self.genColors(i)
            self.updateMeta(i)
            a = {
            "@search.action": "upload",
            "filename": base64.urlsafe_b64encode(i.filename[len(self.file_source)+1:-4]),
            "Camera": i.camera,
            "Lens": i.lens,
            "Date": i.date,
            "colors": c,
            }
            v.append(a)
        d = {"value":v}
        url = self.endpoint + "indexes/images/docs/index" + self.api_version
        response  = requests.post(url, headers=self.headers, json=d)
        index_content = response.json()
        # pprint(index_content)

    def sSearch(self,text):
        url = self.endpoint + "indexes/images/docs" + self.api_version + text
        response  = requests.get(url, headers=self.headers, json=text)
        query = response.json()

        return query
