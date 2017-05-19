from PIL import Image
import numpy as np
import os

class ImageConverter:
    def __init__(self):
        self.resizePixel = 28
    
    def getImageArray(self, infilename):
        img = Image.open(infilename)
        img.load()
        data = np.asarray(img, dtype="int32")
        return data
        
    def doImagePreprocessing(self, filePath):
        for root, dirs, files in os.walk(filePath, topdown=False):
            for name in files:
                filePathToSave = root + '\\output\\'
                ''' Create directory if not exists'''
                directory = os.path.dirname(filePathToSave)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imageFile = os.path.join(root, name)
                print(imageFile)
                if (imageFile.endswith(".jpg") or imageFile.endswith(".png")) == True:
                    cnvFilePath = self.__convertImageToGreyScale(imageFile, filePathToSave)
                    self.__resizeImage(cnvFilePath, filePathToSave)
    
    def __convertImageToGreyScale(self, infilename, pathToSave):
        img = Image.open(infilename).convert('L')
        filename = infilename[infilename.rindex('\\')+1:]
        cnvFilePath = os.path.join(pathToSave, filename)
        img.save(cnvFilePath)
        return cnvFilePath
        
    def __resizeImage(self, infilename, pathToSave):
        filename = infilename[infilename.rindex('\\')+1 : infilename.rindex('.')]
        extension = infilename[infilename.rindex('.'):]
        imageFileName = filename + '_com' + extension   
        img = Image.open(infilename)
        img = img.resize((self.resizePixel, self.resizePixel), Image.ANTIALIAS)
        resizedFilePath = os.path.join(pathToSave, imageFileName)    
        img.save(resizedFilePath)
