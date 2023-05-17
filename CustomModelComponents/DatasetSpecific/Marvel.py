import os
import glob
import csv

marvel_base_image_path_train = 'D:/Downloads/Datasets/marvel_heroes/dataset/train'
marvel_base_image_path_test = 'D:/Downloads/Datasets/marvel_heroes/dataset/test'
dataPath = 'C:/Users/srina/PycharmProjects/MachineLearning/CustomModelComponents/DatasetSpecific/MarvelData/'

def image_path_to_dataset(basePath, mode='train'):
    hero_names = os.listdir(basePath)

    completePath = []
    for name in hero_names:
        currentPath = basePath + '/' + name +'/'
        files = os.listdir(currentPath)
        files = [[currentPath + fname] for fname in files]
        completePath.extend(files)

    outputPath = dataPath + mode + '.csv'
    with open(outputPath, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(completePath)







if __name__ == '__main__':
    image_path_to_dataset(marvel_base_image_path_train, mode='train')
    image_path_to_dataset(marvel_base_image_path_test, mode = 'test')
