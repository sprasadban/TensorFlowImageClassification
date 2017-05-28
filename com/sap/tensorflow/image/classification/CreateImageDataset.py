''' https://www.packtpub.com/books/content/python-image-manipulation'''
from PIL import Image
import os

class CreateImageDataset:
    def __init__(self):
        self.angle_45 = 45
        self.angle_90 = 90
        self.angle_135 = 135
        self.angle_180 = 180
        self.angle_270 = 270
    
    def doImageDistortion(self, filePath):
        for root, dirs, files in os.walk(filePath, topdown=False):
            for name in files:
                filePathToSave = root + '\\output\\'
                ''' Create directory if not exists'''
                directory = os.path.dirname(filePathToSave)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                imageFile = os.path.join(root, name)
                if (imageFile.endswith(".jpg") or imageFile.endswith(".png")) == True:
                    self.__createMoreImages(imageFile, filePathToSave)
    
    def __createMoreImages(self, infilename, pathToSave):
        print(infilename)
        img = Image.open(infilename)
        fullFilename = infilename[infilename.rindex('\\')+1:]
        filename = fullFilename[:fullFilename.rindex('.')]
        extension = fullFilename[fullFilename.rindex('.'):]
        for step in range(0, 100, 5):
            degree = step + 45
            print(degree)
            img_step = img.rotate(degree, Image.BICUBIC);
            cnvFilePath = os.path.join(pathToSave, filename +'_c' + str(step) + extension)
            img_step.save(cnvFilePath)
        
        
if __name__ == '__main__':
    #"C:\\D\\Fruits_Dataset\\FIDS30\\"
    filePath = "C:\\D\\Fruits_Dataset\\Test_Distortion"
    createImages = CreateImageDataset()
    createImages.doImageDistortion(filePath);