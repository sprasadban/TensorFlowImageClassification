from com.sap.tensorflow.image.classification.ImageConverter import ImageConverter
from com.sap.tensorflow.image.classification.ImageSelector import ImageSelector

class FruitsImageClassifier:
    def __init__(self):
        self.preprocessedDataSets = {}
        self.imageDataSets = {}
        self.currentBatchSize = 0
    
    def __initializeImageArray(self, imageMap):
        imageData = []
        labels = []
        for key in imageMap.keys():
            values = imageMap.get(key)
            imageFiles = values[0]
            for i in range(len(imageFiles)):
                imageData.append([])
                labels.append([])
        return (imageData, labels)
        
    '''Get Image Array based on Image file'''
    def __getImageTuple(self, imageMap):
        data = self.__initializeImageArray(imageMap)
        imageData = data[0]
        labels = data[1]
        imageTuple = ()
        index = 0    
        for key in imageMap.keys():
            values = imageMap.get(key)
            imageFiles = values[0]
            classLabel = values[1]
            for imageFile in imageFiles:
                fileName = imageFile[:imageFile.rindex('.')]
                extension = imageFile[imageFile.rindex('.'):]
                imageFileName =  key + "\\output\\" + fileName + '_com' + extension
                imageDataMatrix = self.imageConverter.getImageArray(imageFileName)
                imageData[index] = imageDataMatrix.flatten()
                labels[index] = classLabel
                index = index + 1
        imageTuple = (imageData, labels)
        return imageTuple
    
    def __getImageDataset(self):
        dataSets = {}
        dataSets['training'] = self.__getImageTuple(self.preprocessedDataSets['training'])
        #dataSets['validation'] = self.__getImageTuple(self.preprocessedDataSets['validation'])
        dataSets['test'] = self.__getImageTuple(self.preprocessedDataSets['test'])
        return dataSets
     
    def preProcessImages(self, filePath):
        self.imageSelector = ImageSelector()    
        self.imageConverter = ImageConverter()
        '''Select random images using stratified sampling'''
        self.preprocessedDataSets = self.imageSelector.prepareDataSet(filePath)
        '''Convert Images to GreyScale and compress to 28 * 28 size'''
        self.imageConverter.doImagePreprocessing(filePath)
        '''Prepare image dataset array'''
        self.imageDataSets = self.__getImageDataset()
        return self.imageDataSets
    
    def getNextTrainingBatch(self, requestedSize):
        trainingData = self.imageDataSets['training']
        trainingDataSize = len(trainingData[0])
        if(requestedSize > trainingDataSize):
            return trainingData
        elif((self.currentBatchSize < trainingDataSize) and (requestedSize < (trainingDataSize - self.currentBatchSize))):
            imageTuple = self.__getImageData(trainingData, self.currentBatchSize, self.currentBatchSize + requestedSize)
            self.currentBatchSize = self.currentBatchSize + requestedSize
            return imageTuple
        else:
            '''TODO: Shuffle and return first batch'''
            self.currentBatchSize = 0
            imageTuple = self.__getImageData(trainingData, self.currentBatchSize, self.currentBatchSize + requestedSize)
            self.currentBatchSize = self.currentBatchSize + requestedSize
            return trainingData       
    
    def __getImageData(self, trainingData, start, end):
        images = trainingData[0]
        labels = trainingData[1]
        imageTuple = (images[start:end], labels[start:end])
        return imageTuple

            
if __name__ == '__main__':
    filePath = 'C:\\D\\Fruits_Dataset\\FIDS30\\'
    imageClassifier = FruitsImageClassifier()
    imageDataSets = imageClassifier.preProcessImages(filePath);
    print(imageDataSets['training'])
    #print(imageDataSets['validation'])
    print(imageDataSets['test'])
    print(imageClassifier.getNextTrainingBatch(20))
    print(imageClassifier.getNextTrainingBatch(20))
    