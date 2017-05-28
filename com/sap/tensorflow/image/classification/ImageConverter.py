from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np
import os
from tensorflow.contrib.slim.python.slim.data import dataset

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
                if (imageFile.endswith(".jpg") or imageFile.endswith(".png")) == True:
                    cnvFilePath = self.__convertImageToGreyScale(imageFile, filePathToSave)
                    self.__resizeImage(cnvFilePath, filePathToSave)
                    
    def createMoreImages(self, filePath):
        for root, dirs, files in os.walk(filePath, topdown=False):
            if(root.find("\output") != -1):
                continue                
            for name in files:
                filePathToSave = root + '\\output\\'
                ''' Create directory if not exists'''
                directory = os.path.dirname(filePathToSave)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imageFile = os.path.join(root, name)
                if (imageFile.endswith(".jpg") or imageFile.endswith(".png")) == True:
                    cnvFilePath = self.__convertImageToGreyScale(imageFile, filePathToSave)
                    self.__resizeImage(cnvFilePath, filePathToSave)
        
    def AddFlip_Blur_N_Rotation(self, dataSet):
        addedDataset = {}
        for key in dataSet.keys():
            values = dataSet.get(key)
            imageFiles = values[0]
            classLabel = values[1]
            '''Add additional files by flipping, rotating and blurring'''
            addCnvFileNames = []
            for imageFile in imageFiles:
                imageFileName =  key + "\\output\\" + imageFile
                addCnvFileNames.extend(self._flipNRotate(imageFileName))
            '''Add dataset with additional converted image file names'''
            imageFiles.extend(addCnvFileNames)
            addedDataset[key] = (imageFiles, classLabel)
        return addedDataset
                
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

    def _flipNRotate(self, infilename):
        savedFileNames = []
        im = Image.open(infilename)
    
        filePathToSave = infilename[:infilename.rindex('\\')]  
        ''' Create directory if not exists'''
        directory = os.path.dirname(filePathToSave)
        if not os.path.exists(directory):
            os.makedirs(directory)    
        filename = infilename[infilename.rindex('\\')+1 : infilename.rindex('.')]
        extension = infilename[infilename.rindex('.'):]
        
        '''self._filter(im, ImageFilter.DETAIL, filename, '_DETAIL', extension, filePathToSave, savedFileNames)
        self._filter(im, ImageFilter.SHARPEN, filename, '_SHARPEN', extension, filePathToSave, savedFileNames)        
        self._filter(im, ImageFilter.SMOOTH, filename, '_SMOOTH', extension, filePathToSave, savedFileNames)
        self._filter(im, ImageFilter.SMOOTH_MORE, filename, '_SMOOTH_MORE', extension, filePathToSave, savedFileNames)
        self._filter(im, ImageFilter.BLUR, filename, '_BLUR', extension, filePathToSave, savedFileNames)'''
        
        self._transpose(im, Image.FLIP_LEFT_RIGHT, filename, '_FLIP_LEFT_RIGHT', extension, filePathToSave, savedFileNames)
        '''self._transpose(im, Image.FLIP_TOP_BOTTOM, filename, '_FLIP_TOP_BOTTOM', extension, filePathToSave, savedFileNames)
        self._transpose(im, Image.ROTATE_90, filename, '_ROTATE_90', extension, filePathToSave, savedFileNames)
        self._transpose(im, Image.ROTATE_180, filename, '_ROTATE_180', extension, filePathToSave, savedFileNames)
        self._transpose(im, Image.ROTATE_270, filename, '_ROTATE_270', extension, filePathToSave, savedFileNames)
        self._transpose(im, Image.TRANSPOSE, filename, '_TRANSPOSE', extension, filePathToSave, savedFileNames)'''
        
        '''self._contrast(im, 1.05, filename, '_05CONTRAST', extension, filePathToSave, savedFileNames);
        self._contrast(im, 1.1, filename, '_10CONTRAST', extension, filePathToSave, savedFileNames);
        self._contrast(im, 1.15, filename, '_15CONTRAST', extension, filePathToSave, savedFileNames);
        self._contrast(im, 1.20, filename, '_20CONTRAST', extension, filePathToSave, savedFileNames);
        self._contrast(im, 1.25, filename, '_25CONTRAST', extension, filePathToSave, savedFileNames);
        self._contrast(im, 1.30, filename, '_30CONTRAST', extension, filePathToSave, savedFileNames);'''
        
        return savedFileNames
    

    def _filter(self, image, filterType, filename, suffix, extension, filePathToSave, savedFileNames):
        out = image.filter(filterType)
        imageFileName = filename + suffix + extension
        savedFileNames.append(imageFileName);  
        self._saveImageNCompress(out, filePathToSave, imageFileName)       
    
    def _transpose(self, image, flipType, filename, suffix, extension, filePathToSave, savedFileNames):
        out = image.transpose(flipType)
        imageFileName = filename + suffix + extension
        savedFileNames.append(imageFileName);  
        self._saveImageNCompress(out, filePathToSave, imageFileName)       

    def _contrast(self, image, enhancer, filename, suffix, extension, filePathToSave, savedFileNames):
        enh = ImageEnhance.Contrast(image)
        out = enh.enhance(enhancer)
        imageFileName = filename + suffix + extension
        savedFileNames.append(imageFileName);       
        self._saveImageNCompress(out, filePathToSave, imageFileName)       
        
    def _saveImageNCompress(self, image, filePathToSave, imageFileName):
        cnvFilePath = os.path.join(filePathToSave, imageFileName)    
        image.save(cnvFilePath)
        self.__resizeImage(cnvFilePath, filePathToSave)

if __name__ == '__main__':
    filePath = 'C:\\Supermarket_Produce_Dataset\\Fruits'
    imageCnv = ImageConverter()
    imageCnv.createMoreImages(filePath);         