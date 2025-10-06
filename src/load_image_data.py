import os
import json
import pandas as pd
from tqdm import tqdm

from labels import (
    get_cifar10_labels, get_cifar10objects_labels,
    get_cifar10animals_labels, get_food101_labels,
    get_cifar100_labels, get_country211_labels,
    get_mnist_labels, get_inaturalist_labels,
    get_fmnist_labels, get_oxfordpets_labels,
    get_imageneta_labels, get_imagenetr_labels,
    get_fairface_labels, get_socialcounterfactuals_labels
)

from labels import (
    get_occupations,
    get_data_paths
)

def get_label_names(dataset, attribute=None, is_joined=False):
    image_classes = {}
    image_classes['cifar10'] = get_cifar10_labels(is_joined)
    image_classes['cifar10animals'] = get_cifar10animals_labels(is_joined)
    image_classes['cifar10objects'] = get_cifar10objects_labels(is_joined)
    image_classes['food101'] = get_food101_labels()
    image_classes['cifar100'] = get_cifar100_labels()
    image_classes['country211'] = get_country211_labels(is_joined)
    image_classes['mnist'] = get_mnist_labels(is_joined)
    image_classes['inaturalist'] = get_inaturalist_labels(is_joined)
    image_classes['fmnist'] = get_fmnist_labels(is_joined)
    image_classes['oxfordpets'] = get_oxfordpets_labels(is_joined)
    image_classes['imagenet-a'] = get_imageneta_labels(is_joined)
    image_classes['imagenet-r'] = get_imagenetr_labels(is_joined)
    image_classes['fairface-class'] = get_fairface_labels(attribute, is_joined)
    image_classes['socialcounterfactuals-class'] = get_socialcounterfactuals_labels(attribute, is_joined)
    return image_classes[dataset]

def get_images(dataset, is_joined=False):
    image_path = get_data_paths(dataset)
    labels = get_label_names(dataset, is_joined)
    image_labels = {str(i): c for i, c in enumerate(labels)}

    if(dataset == 'country211'):
        df = pd.read_excel(os.path.join(image_path, 'data.xlsx'))
        df = df.astype(str)
        iso_labels = {}
        for ii in range(len(df)):
            if(df['Name'][ii] == 'Namibia'):
                iso_labels[df['Name'][ii]] = 'NA'
            else:
                iso_labels[df['Name'][ii]] = df['Code'][ii]
        iso_labels['Kosovo'] = 'XK'

        new_image_labels = {}
        for i, label in enumerate(labels):
            new_image_labels[i] = iso_labels[label]
        
        #print(new_image_labels)

        image_labels = new_image_labels
        labels = list(image_labels.keys())

    #print(image_labels)
    reverse_image_labels = {c: i for i, c in image_labels.items()}
    #print(reverse_image_labels)

    if(dataset == 'imagenet'):
        image_path = os.path.join(os.path.join(image_path, 'val'))
    elif(dataset in ['cifar10', 'cifar10objects', 'cifar10animals', 'cifar100', 'country211', 'fmnist']):
        image_path = os.path.join(os.path.join(image_path, 'test'))
    elif(dataset in ['mnist']):
        image_path = os.path.join(os.path.join(image_path, 'testing'))

    test_images = []
    test_labels = []
    
    if(dataset in ['cifar10', 'cifar10animals', 'cifar10objects', 'cifar100', 'mnist', 'country211']):
        for id, label in image_labels.items():
            for curimage in os.listdir(os.path.join(image_path, label)):
                test_images.append(os.path.join(image_path, label, curimage))
                test_labels.append(int(id))
    elif dataset in ['food101']:
        with open(os.path.join(image_path, 'meta', 'test.txt'), 'r') as f:
            for line in f.readlines():
                test_images.append(os.path.join(image_path, 'images', line.strip() + '.jpg'))
                curlabel = line.strip().split("/")[0]
                test_labels.append(int(reverse_image_labels[curlabel]))
    elif dataset in ['inaturalist']:
        with open(os.path.join(image_path, 'val2017.json'), 'r') as f:
            json_file = json.load(f)

        for cur_image in json_file['images']:
            test_images.append(os.path.join(image_path, cur_image['file_name']))
            curlabel = line.strip().split("/")[0]
            test_labels.append(int(reverse_image_labels[curlabel]))
    elif dataset in ['fmnist']:
        for curimage in os.listdir(image_path):
            test_images.append(os.path.join(image_path, curimage))
            test_labels.append(int(curimage.split('_')[0]))
    elif dataset in ['oxfordpets']:
        dogs = ['american', 'basset', 'beagle', 'boxer', 'chihuahua', 'english', 'german', 'great', 'havanese', 
                'japanese', 'keeshond', 'leonberger', 'miniature', 'newfoundland', 'pomeranian', 'pug', 'saint', 
                'samoyed', 'scottish', 'shiba', 'staffordshire', 'wheaten', 'yorkshire']
        cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British', 'Egyptian', 'Maine', 'Persian', 'Ragdoll', 
                'Russian', 'Siamese', 'Sphynx']
        for curimage in os.listdir(os.path.join(image_path, 'images')):
            if(curimage.endswith('.mat') == True):
                continue

            test_images.append(os.path.join(image_path, 'images', curimage))
            test_labels.append(int(labels.index(curimage.split('_')[0])))
    elif dataset in ['imagenet-a', 'imagenet-r']:
        synset_to_class = {}
        with open(os.path.join(image_path, 'README.txt'), 'r') as f:
            for line in f.readlines():
                synset = line.strip().split()[0]

                if(dataset == 'imagenet-a'):
                    synset_to_class[synset] = " ".join(line.strip().split()[1:])
                else:
                    synset_to_class[synset] = " ".join(line.strip().split()[1].split("_"))
        
        for synset_folder in os.listdir(image_path):
            if(synset_folder.endswith('.txt')):
                continue
            for cur_folder in os.listdir(os.path.join(image_path, synset_folder)):
                test_images.append(os.path.join(image_path, synset_folder, cur_folder))
                test_labels.append(labels.index(synset_to_class[synset_folder]))

    return test_images, test_labels

def get_fairness_dataset(dataset, attribute, is_joined=False):
    datapath = get_data_paths(dataset)

    races = ['asian', 'black', 'indian', 'latino', 'middle eastern', 'white']
    genders = ['male', 'female']

    if(dataset == 'socialcounterfactuals-skew'):
        if(attribute == 'occupation-gender'):
            images = []
            occupations = get_occupations(dataset)
            df = pd.read_csv(os.path.join(datapath, 'metadata.csv'))
            for occindex in tqdm(range(len(occupations))):
                eachoccupation = occupations[occindex]
                occupation_list = [[]] * len(genders)
                
                for ii in range(len(df)):
                    cur_gender = df['a2'][ii].lower()
                    if(cur_gender not in genders):
                        continue

                    if(df['file_name'][ii].split('_')[-4] == eachoccupation):
                        gender_idx = genders.index(cur_gender)
                        occupation_list[gender_idx].append(os.path.join(datapath, df['file_name'][ii]))

                images.append(occupation_list)

            return images
        elif(attribute == 'occupation-race'):
            images = []
            occupations = get_occupations(dataset)
            df = pd.read_csv(os.path.join(datapath, 'metadata.csv'))
            for occindex in tqdm(range(len(occupations))):
                eachoccupation = occupations[occindex]
                occupation_list = [[]] * len(races)
                
                for ii in range(len(df)):
                    cur_race = df['a1'][ii].lower()
                    if(cur_race not in races):
                        continue

                    if(df['file_name'][ii].split('_')[-4] == eachoccupation):
                        race_idx = races.index(cur_race)
                        occupation_list[race_idx].append(os.path.join(datapath, df['file_name'][ii]))

                images.append(occupation_list)

            return images
    elif(dataset == 'fairface-skew'):
        datapath = get_data_paths(dataset)
        df = pd.read_csv(os.path.join(datapath, 'fairface_label_val.csv'))
        if(attribute == 'race'):
            images = []
            for ii in range(len(df)):
                cur_race = df['race'][ii]
                if(cur_race == 'East Asian' or cur_race == 'Southeast Asian'):
                    cur_race = cur_race.split()[1]
                elif(cur_race == 'Latino_Hispanic'):
                    cur_race = cur_race.split('_')[0]
                
                cur_race = cur_race.lower()
                images.append((os.path.join(datapath, df['file'][ii]), cur_race, races.index(cur_race)))
            
            return images

        elif(attribute == 'gender'):
            images = []
            for ii in range(len(df)):
                cur_gender = df['gender'][ii]
                cur_gender = cur_gender.lower()
                images.append((os.path.join(datapath, df['file'][ii]), cur_gender, genders.index(cur_gender)))
            
            return images
        
    elif(dataset == 'fairface-class'):
        datapath = get_data_paths(dataset)
        df = pd.read_csv(os.path.join(datapath, 'fairface_label_val.csv'))
        
        if(attribute == 'race'):
            images = []
            labels = []
            for ii in range(len(df)):
                cur_race = df['race'][ii]
                if(cur_race == 'East Asian' or cur_race == 'Southeast Asian'):
                    cur_race = cur_race.split()[1]
                elif(cur_race == 'Latino_Hispanic'):
                    cur_race = cur_race.split('_')[0]
                
                cur_race = cur_race.lower()
                images.append(os.path.join(datapath, df['file'][ii]))
                labels.append(races.index(cur_race))
            
            return images, labels
        elif(attribute == 'gender'):
            images = []
            labels = []
            for ii in range(len(df)):
                cur_gender = df['gender'][ii]
                
                cur_gender = cur_gender.lower()
                images.append(os.path.join(datapath, df['file'][ii]))
                labels.append(genders.index(cur_gender))
            
            return images, labels
    elif(dataset == 'socialcounterfactuals-class'):
        df = pd.read_csv(os.path.join(datapath, 'metadata.csv'))
        if(attribute == 'race'):
            images = []
            labels = []
            for ii in range(len(df)):
                cur_race = df['a1'][ii].lower()
                if(cur_race not in races):
                    continue
                
                cur_race = cur_race.lower()
                images.append(os.path.join(datapath, df['file_name'][ii]))
                labels.append(races.index(cur_race))
            
            return images, labels
        elif(attribute == 'gender'):
            images = []
            labels = []
            for ii in range(len(df)):
                cur_gender = df['a2'][ii].lower()
                if(cur_gender not in genders):
                    continue
                
                cur_gender = cur_gender.lower()
                images.append(os.path.join(datapath, df['file_name'][ii]))
                labels.append(genders.index(cur_gender))
            
            return images, labels
    '''
    if(attribute == 'race'):
        if(dataset == 'fairface'):
            df = pd.read_csv(os.path.join(datapath, 'fairface_label_train.csv'))
            images = {}
            for ii in range(len(df)):
                cur_race = df['race'][ii]
                if(cur_race == 'East Asian' or cur_race == 'Southeast Asian'):
                    cur_race = cur_race.split()[1]
                elif(cur_race == 'Latino_Hispanic'):
                    cur_race = cur_race.split('_')[0]
                
                cur_race = cur_race.lower()
                if(cur_race in images):
                    images[cur_race].append(os.path.join(datapath, df['file'][ii]))
                else:
                    images[cur_race] = [os.path.join(datapath, df['file'][ii])]
            
            return images
        elif(dataset == 'socialcounterfactuals'):
            df = pd.read_csv(os.path.join(datapath, 'metadata.csv'))
            images = {}
            for ii in range(len(df)):
                cur_race = df['a1'][ii].lower()

                if(cur_race not in races):
                    continue

                if(cur_race in images):
                    images[cur_race].append(os.path.join(datapath, df['file_name'][ii]))
                else:
                    images[cur_race] = [os.path.join(datapath, df['file_name'][ii])]

            new_images = {}
            for key, val in images.items():
                new_images[key] = val
            
            return new_images
            #return images
    elif(attribute == 'gender'):
        if(dataset == 'fairface'):
            df = pd.read_csv(os.path.join(datapath, 'fairface_label_train.csv'))
            images = {}
            for ii in range(len(df)):
                cur_race = df['race'][ii]
                if(cur_race == 'East Asian' or cur_race == 'Southeast Asian'):
                    cur_race = cur_race.split()[1]
                elif(cur_race == 'Latino_Hispanic'):
                    cur_race = cur_race.split('_')[0]
                
                cur_race = cur_race.lower()
                if(cur_race in images):
                    images[cur_race].append(os.path.join(datapath, df['file'][ii]))
                else:
                    images[cur_race] = [os.path.join(datapath, df['file'][ii])]
            
            return images
        elif(dataset == 'socialcounterfactuals'):
            df = pd.read_csv(os.path.join(datapath, 'metadata.csv'))
            images = {}
            for ii in range(len(df)):
                cur_gender = df['a2'][ii].lower()

                if(cur_gender not in genders):
                    continue

                if(cur_gender in images):
                    images[cur_gender].append(os.path.join(datapath, df['file_name'][ii]))
                else:
                    images[cur_gender] = [os.path.join(datapath, df['file_name'][ii])]


            new_images = {}
            for key, val in images.items():
                new_images[key] = val
            
            return new_images
            #return images
    '''