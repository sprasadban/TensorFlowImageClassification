from com.sap.tensorflow.image.classification.ImageConverter import ImageConverter
from com.sap.tensorflow.image.classification.ImageSelector import ImageSelector
from _warnings import filters

class FruitsImageClassifier:
    def __init__(self, image_count):
        self.preprocessedDataSets = {}
        self.imageDataSets = {}
        self.currentBatchSize = 0
        self.imageCount = image_count
    
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
    def __getImageTuple(self, imageMap, output, filters):
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
                if(len(filters) == 0 and output == False):
                    fileName = imageFile[:imageFile.rindex('.')]
                    extension = imageFile[imageFile.rindex('.'):]
                    imageFileName =  key + "\\output\\" + fileName + "_com" + extension
                else:
                    for filter in filters:
                        if(imageFile.find(filter) != -1):
                            imageFileName =  key + "\\" + imageFile
                            break;
                if(imageFileName != ""):
                    imageDataMatrix = self.imageConverter.getImageArray(imageFileName)
                    '''Flatten N-D Matrix to 1-D Array'''
                    imageData[index] = imageDataMatrix.flatten()
                    labels[index] = classLabel
                    index = index + 1
        imageTuple = (imageData, labels)
        return imageTuple
    
    def __getImageDataset(self, output= False, filterName="_com"):
        dataSets = {}
        dataSets['training'] = self.__getImageTuple(self.preprocessedDataSets['training'], output, filterName)
        #dataSets['validation'] = self.__getImageTuple(self.preprocessedDataSets['validation'])
        dataSets['test'] = self.__getImageTuple(self.preprocessedDataSets['test'], output, filterName)
        return dataSets
     
    def preProcessImages(self, filePath):
        self.imageSelector = ImageSelector(self.imageCount)    
        self.imageConverter = ImageConverter()
        '''Select random images using stratified sampling'''
        self.preprocessedDataSets = self.imageSelector.prepareDataSet(filePath)
        '''Convert Images to GreyScale and compress to 28 * 28 size'''
        self.imageConverter.doImagePreprocessing(filePath)
        '''Prepare image dataset array'''
        self.imageDataSets = self.__getImageDataset()
        return self.imageDataSets

    def preProcessRandomImages(self, filePath):
        self.imageSelector = ImageSelector(self.imageCount)    
        self.imageConverter = ImageConverter()
        '''Select random images using stratified sampling'''
        self.preprocessedDataSets = self.imageSelector.prepareRandomDataSet(filePath)
        '''Convert Images to GreyScale and compress to 28 * 28 size'''
        self.imageConverter.doImagePreprocessing(filePath)
        '''Add Create extra synthetic training data by flipping, rotating and blurring the images on our data set'''
        trainingDataset = self.preprocessedDataSets['training']
        self.preprocessedDataSets['training'] = self.imageConverter.AddFlip_Blur_N_Rotation(trainingDataset)
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
            '''Get remaining images and reset starting batch point'''
            imageTuple = self.__getImageData(trainingData, self.currentBatchSize, trainingDataSize)
            self.currentBatchSize = 0
            return imageTuple       
    
    def __getImageData(self, trainingData, start, end):
        images = trainingData[0]
        labels = trainingData[1]
        imageTuple = (images[start:end], labels[start:end])
        return imageTuple
    
    def loadPreprocessedImages(self, filePath, filters):
        self.imageSelector = ImageSelector(self.imageCount) 
        self.imageConverter = ImageConverter()
        self.preprocessedDataSets = self.imageSelector.getPreprocessedImages(filePath, filters) 
        '''Prepare image dataset array'''
        self.imageDataSets = self.__getImageDataset(True, filters)
        return self.imageDataSets       
            
if __name__ == '__main__':
    filePath = 'C:\\Supermarket_Produce_Dataset\\tropical-fruits-DB-1024x768'
    imageClassifier = FruitsImageClassifier(264)
    #imageDataSets = imageClassifier.preProcessImages(filePath);
    imageDataSets = imageClassifier.preProcessRandomImages(filePath);
    print(imageDataSets['training'])
    #print(imageDataSets['validation'])
    print(imageDataSets['test'])
    print(imageClassifier.getNextTrainingBatch(20))
    print(imageClassifier.getNextTrainingBatch(20))
    