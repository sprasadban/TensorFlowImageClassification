from PIL import Image
import os
from PIL import ImageEnhance
from PIL import ImageFilter

def flipNRotate(infilename):
    im = Image.open(infilename)
    im.load()

    filePathToSave = infilename[:infilename.rindex('\\')] + '\\output\\'  
    print(filePathToSave)      
    ''' Create directory if not exists'''
    directory = os.path.dirname(filePathToSave)
    if not os.path.exists(directory):
        os.makedirs(directory)    
    filename = infilename[infilename.rindex('\\')+1 : infilename.rindex('.')]
    extension = infilename[infilename.rindex('.'):]
    
    out = im.filter(ImageFilter.BLUR)
    imageFileName = filename + '_BLUR' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.CONTOUR)
    imageFileName = filename + '_CONTOUR' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.DETAIL)
    imageFileName = filename + '_DETAIL' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.EDGE_ENHANCE)
    imageFileName = filename + '_EDGE_ENHANCE' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
    imageFileName = filename + '_EDGE_ENHANCE_MORE' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.EMBOSS)
    imageFileName = filename + '_EMBOSS' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.FIND_EDGES)
    imageFileName = filename + '_FIND_EDGES' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.SMOOTH)
    imageFileName = filename + '_SMOOTH' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.SMOOTH_MORE)
    imageFileName = filename + '_SMOOTH_MORE' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.filter(ImageFilter.SHARPEN)
    imageFileName = filename + '_SHARPEN' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    out = im.transpose(Image.FLIP_LEFT_RIGHT)
    imageFileName = filename + '_FLIP_LEFT_RIGHT' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    out = im.transpose(Image.FLIP_TOP_BOTTOM)
    imageFileName = filename + '_FLIP_TOP_BOTTOM' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    out = im.transpose(Image.ROTATE_90)
    imageFileName = filename + '_ROTATE_90' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    out = im.transpose(Image.ROTATE_180)
    imageFileName = filename + '_ROTATE_180' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    out = im.transpose(Image.ROTATE_270)
    imageFileName = filename + '_ROTATE_270' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    out = im.transpose(Image.TRANSPOSE)
    imageFileName = filename + '_TRANSPOSE' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.05)
    imageFileName = filename + '_05CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.1)
    imageFileName = filename + '_10CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.15)
    imageFileName = filename + '_15CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.20)
    imageFileName = filename + '_20CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.25)
    imageFileName = filename + '_25CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)

    enh = ImageEnhance.Contrast(im)
    out = enh.enhance(1.30)
    imageFileName = filename + '_30CONTRAST' + extension       
    cnvFilePath = os.path.join(filePathToSave, imageFileName)    
    out.save(cnvFilePath)
    
if __name__ == '__main__':
    flipNRotate('C:\\D\\Fruits_Dataset\\Flip\\8.jpg')
