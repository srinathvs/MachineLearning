import os.path
import os
import sys

import numpy as np

import glob
import csv

datasets_base_path = "D:\\Downloads\\Datasets\\"
pokemon_base_path = datasets_base_path + "Pokemon_Complete_Dataset"
pokemon_images_base_path = pokemon_base_path + '\\images'

created_dataset_path = "C:\\Users\\srina\\PycharmProjects\\MachineLearning\\CustomModelComponents\\DatasetSpecific\\PokemonData\\"


def convert_pokemon_type_to_label(type_dict, pokemon_type1, pokemon_type2):
    index_number1 = type_dict[pokemon_type1]
    index_number2 = type_dict[pokemon_type2]
    base_array = np.zeros_like(list(type_dict.keys()), dtype=int)
    base_array[index_number1] = 1
    base_array[index_number2] = 1
    return base_array


def create_pokemon_dataset_csv(csv_path=pokemon_base_path + '\\pokemon.csv'):
    pokemon_dict = {}
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file, dialect='excel')

        for row in csv_reader:
            if len(row) > 1:
                pokemon_name = row[0]
                pokemon_type1 = row[1]
                if len(row) == 3:
                    pokemon_type2 = row[2]
                else:
                    pokemon_type2 = pokemon_type1

                pokemon_images = sorted(list(glob.glob(pokemon_images_base_path + '\\' + pokemon_name + '\\*')))

                if len(pokemon_images) > 0:
                    for image_path in pokemon_images:
                        pokemon_dict[image_path] = (pokemon_name, pokemon_type1, pokemon_type2)

    print(len(pokemon_dict))
    type_dict = {}
    item = 0
    for links, tuple_element in list(pokemon_dict.items()):
        name, type1, type2 = tuple_element
        if type1 not in type_dict.keys():
            type_dict[type1] = item
            item += 1
            if type2 not in type_dict.keys():
                type_dict[type2] = item
                item += 1
        elif type2 not in type_dict.keys():
            type_dict[type2] = item
            item += 1

    pokemon_list = []
    for links, tuple_element in pokemon_dict.items():
        name, type1, type2 = tuple_element
        type_label = convert_pokemon_type_to_label(type_dict, type1, type2)
        pokemon_list.append([links, type_label])

    print(pokemon_list[:5])

    type_list = []
    for type_name in type_dict.keys():
        type_list.append([type_name, type_dict[type_name]])

    with open(created_dataset_path + 'type_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(type_list)

    with open(created_dataset_path + 'pokemon_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(pokemon_list)


def recreate_type_dict():
    type_dict = {}
    with open(created_dataset_path + 'type_data.csv', 'r') as f:
        type_reader = csv.reader(f, dialect='excel')
        for row in type_reader:
            type_dict[row[0]] = row[1]
    return type_dict


if __name__ == '__main__':
    create_pokemon_dataset_csv()
