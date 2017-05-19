import random
import os
from builtins import int

class ImageSelector:
    def __init__(self):
        self.classCount = 10
        self.population = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
        self.trainingPositions = []
        #self.validationPositions = []
        self.testPositions = []
        self.allImages = {}
        self.trainingSet = {}
        #self.validationSet = {}
        self.testSet = {}
        '''80% Training set'''        
        self.trainingLen = int(len(self.population) * 0.8)
        '''15% Validation set'''                
        #self.validationLen = int(len(self.population) * 0.1)
        '''20% Test set'''        
        self.testLen = int(len(self.population) * 0.2)
    
    def __randomGenerator(self):
        random.shuffle(self.population)
        self.trainingPositions = self.population[:self.trainingLen]
        #self.validationPositions = self.population[self.trainingLen:(self.trainingLen + self.validationLen)]
        #self.testPositions = self.population[(self.trainingLen + self.validationLen):(self.trainingLen + self.validationLen + self.testLen)]
        self.testPositions = self.population[self.trainingLen:(self.trainingLen + self.testLen)]
    
    def __getClassLabel(self, position):
        classLabel = []
        for i in range(self.classCount):
            classLabel.append(0)
        classLabel[position] = 1
        return classLabel
        
    def __getAllImages(self, filePath):
        rootChange = ""
        imageFiles = []
        position = 0
        for root, dirs, files in os.walk(filePath, topdown=False):
            if(rootChange != root):
                if(rootChange != ''):
                    '''Map containing key as 'folder_name' and values are tuple with (image_file, class_label)'''
                    self.allImages[rootChange] = (imageFiles, self.__getClassLabel(position))
                    position = position + 1
                rootChange = root
                imageFiles = []
            for name in files:
                imageFiles.append(name)
        return self.allImages
    
    def __getTrainingSetImages(self, imageFiles):
        trainingImages = []
        for i in self.trainingPositions:
            trainingImages.append(imageFiles[i])
        return trainingImages

    def __getValidationSetImages(self, imageFiles):
        validationImages = []
        for i in self.validationPositions:
            validationImages.append(imageFiles[i])
        return validationImages
    
    def __getTestSetImages(self, imageFiles):
        testingImages = []
        for i in self.testPositions:
            testingImages.append(imageFiles[i])
        return testingImages
            
    def prepareDataSet(self, filePath):
        datasets = {}
        '''Random genertion of population set'''
        self.__randomGenerator()
        '''Remove all output folders under image folders'''
        self.__cleanUp(filePath)
        '''Get images dataset using stratified sampling'''
        allImages = self.__getAllImages(filePath)
        for key in allImages.keys():
            values = allImages.get(key)
            imageFiles = values[0]
            classLabel = values[1]            
            '''Training set images based on positions'''
            self.trainingSet[key] = (self.__getTrainingSetImages(imageFiles), classLabel)
            '''Validation set images based on positions'''        
            #self.validationSet[key] = (self.__getValidationSetImages(imageFiles), classLabel)
            '''Test set images based on positions'''        
            self.testSet[key] = (self.__getTestSetImages(imageFiles), classLabel)
        datasets['training'] = self.trainingSet
        #datasets['validation'] = self.validationSet
        datasets['test'] = self.testSet
        return datasets
    
    def __cleanUp(self, filePath):
        for root, dirs, files in os.walk(filePath, topdown=False):
            for filename in files:
                if('output' in root):
                    os.remove(os.path.join(root, filename))
            if('output' in root):
                os.rmdir(root)


if __name__ == '__main__':
    filePath = 'C:\\D\\Fruits_Dataset\\FIDS30\\'
    imageFileSelector = ImageSelector()
    dataSets = imageFileSelector.prepareDataSet(filePath)
    print(dataSets['training'])
    #print(dataSets['validation'])
    print(dataSets['test'])
    