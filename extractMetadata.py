import subprocess
import datetime

from threading import Thread

class extractor(object):

    def __init__(self,fFrom):
        """
            data_list contains dictionaries for each photograph, this contains
            output metadata recovered

            list_stripped contains fairly useless metadata stripped from data_list

            file_source is the folder source of files

            write_to is where to write them

        """
        self.data_dict = {}
        self.list_stripped = []
        self.file_source = fFrom

    def formatTime(self,t):
        lTimeS = t["Date/Time Original"].replace(":"," ").replace("."," ").split()
        lTime = [int(a) for a in lTimeS]
        lTime[-1] *= 10000

        d1 = datetime.datetime(*lTime)
        return d1


    def extract_metadata(self):
        metadataraw = subprocess.check_output(['./Image-ExifTool-11.28/exiftool',self.file_source])
        rawstr = metadataraw.decode('ASCII',errors="ignore") #turning the byte string to ASCII
        rawstr = rawstr.split('========')
        imageData = [i.split('\n') for i in rawstr]
        for files in imageData:
            files.pop(0)
        for items in imageData:
            tempDict = {}
            for fields in items:
                fieldsList = fields.split(": ",1)
                if (len(fieldsList) > 1):
                    tempDict[fieldsList[0].rstrip()]=fieldsList[1]
            if (tempDict != {}):
            	self.data_dict[tempDict['File Name']] = dict(tempDict)
        print("")
      	return self.data_dict
