# coding: utf-8
import pickle
import glob
import os.path
import numpy as np
import json
from PIL import Image
import settings
from nimblenet.data_structures import Instance
setting = settings.imgread_settings
class_to_predict = settings.classes_to_predict

# :param path of the image
# output: an array containing the image in the form of [{
#   label:["cat"]
#   data: [ [RedArray][GreenArray][BlueArray] ]
#  },{
#   label:["horse"]
#   data: [ [RedArray][GreenArray][BlueArray] ]
#  },...
# ]
def readAndNormalizeImg(path):
    ext = os.path.splitext(path)[-1]
    if ext not in setting["img_ext"]:
        if ext != ".json":
            print ext+" is not a know img extention"
        return None
    data  = []
    img = Image.open(path)
    try:
        img.thumbnail(setting["size"])
    except IOError:
        print(path)
        return None
    data = []
    img_data = []
    if setting["black_and_white"]:
        img_data = [np.asarray(img.convert('L'))]
    else:
        img_data = []
        for mono_chroma_img in img.split():
            img_data.append(np.asarray(mono_chroma_img))
    data = []
    col_size = setting["size"][0]
    row_size = setting["size"][1]
    for mono_chroma_img in img_data :
        row_id = 0
        for row in mono_chroma_img:
            dataRow = np.zeros(col_size)
            dataRow[:len(row)] = row
            data.extend(dataRow)
            row_id += 1
        for i in range(row_id,row_size):
            data.extend(np.zeros(col_size))

    data = [float(x) / 255.0 for x in data]
    return data

def createOutputLayerFromMetadata(path):
    print path
    metadata = json.loads(open(path).read())
    layer = []
    for classe in class_to_predict:
        if classe in metadata["labels"]:
            layer.append(1)
        else :
            layer.append(0)
    return layer

def readDataSource(pickle_unpickle = False):
    if pickle_unpickle is not False:
        if os.path.isfile(pickle_unpickle) :
            print("reading "+pickle_unpickle+"...")
            return pickle.load(open(pickle_unpickle,"rb"))
    data = []
    in_out_layer = [] 
    inputLayerSize = setting["size"][0]*setting["size"][1]*(1 if setting["black_and_white"] else 3)
    outputLayerSize = len(class_to_predict)
    directories = glob.glob(setting["data_source"])
    for directory in directories:
        if os.path.isdir(directory):
            outputLayer = createOutputLayerFromMetadata(directory+"/metadata.json")
            for imageFile in glob.glob(directory+"/*"):
                imgData = readAndNormalizeImg(imageFile)
                if imgData is not None:
                    in_out_layer.append(Instance(imgData,outputLayer))
    if pickle_unpickle is not False:
        print("save data to " + pickle_unpickle + "...")
        pickle.dump([inputLayerSize, outputLayerSize, in_out_layer], open(pickle_unpickle, "wb"))
    return inputLayerSize, outputLayerSize, in_out_layer

