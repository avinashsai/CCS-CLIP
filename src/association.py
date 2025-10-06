import random
import numpy as np
random.seed(0)
np.random.seed(0)

def get_entagled_heads(layers_heads):
    all_labels = list(layers_heads.keys())

    result = []
    for ii in range(len(all_labels) - 1, -1, -1):
        label1 = layers_heads[all_labels[ii]]
        for jj in range(ii - 1, -1, -1):
            label2 = layers_heads[all_labels[jj]]
            if(ii == jj):
                continue
            if(label1.startswith(label2) or label2.startswith(label1)):
                if(label1 not in result and label2 not in result):
                    result.append(all_labels[ii])
    
    return list(set(result))

def get_vit_b_32_openai(property_type='high'):
    high_layer_heads = ['L11.H5', 'L10.H8', 'L9.H3', 'L8.H11', 'L9.H2', 'L11.H1', 'L11.H9', 'L11.H7']
    low_layer_heads = ['L8.H5', 'L11.H0', 'L11.H8', 'L9.H9']
    layers = [8, 9, 10, 11]

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'locations'):
        layers_heads = ['L10.H8', 'L11.H9']
    elif(property_type == 'objects'):
        layers_heads = ['L9.H2', 'L11.H1', 'L11.H7']
    elif(property_type == 'colors'):
        layers_heads = ['L11.H5']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(0, 12):
            for head in range(0, 12):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))
    elif(property_type == 'high-last'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-1]):
                layers_heads.append(each_layer_head)
    elif(property_type == 'high-secondlast'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-2]):
                layers_heads.append(each_layer_head)

    return layers_heads

def get_vit_b_32_datacomp(property_type='high'):
    high_layer_heads = ['L9.H3', 'L8.H3', 'L11.H10', 'L9.H10', 'L10.H7', 'L10.H11', 'L8.H1', 'L8.H10', 'L11.H9', 'L11.H4', 'L11.H3']
    low_layer_heads = ['L11.H0', 'L9.H5', 'L11.H11', 'L8.H2', 'L11.H1', 'L8.H4', 'L9.H4', 'L8.H9']

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'locations'):
        layers_heads = ['L10.H7']
    elif(property_type == 'objects'):
        layers_heads = ['L8.H1', 'L8.H10', 'L9.H10', 'L10.H11', 'L11.H10']
    elif(property_type == 'colors'):
        layers_heads = ['L11.H3', 'L11.H4', 'L11.H9']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(8, 12):
            for head in range(0, 12):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))

    return layers_heads

def get_vit_b_16_openai(property_type='high'):
    high_layer_heads = ['L11.H0', 'L10.H5', 'L8.H8', 'L10.H7', 'L11.H6', 'L8.H5', 'L11.H4', 'L11.H3', 'L11.H7', 'L11.H11']
    low_layer_heads = ['L10.H6', 'L11.H2', 'L9.H3', 'L8.H0']

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'animals'):
        layers_heads = ['L11.H11']
    elif(property_type == 'locations'):
        layers_heads = ['L11.H6']
    elif(property_type == 'colors'):
        layers_heads = ['L11.H7']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(8, 12):
            for head in range(0, 12):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))

    return layers_heads

def get_vit_b_16_laion(property_type='high'):
    high_layer_heads = ['L11.H0', 'L9.H0', 'L9.H3', 'L10.H5', 'L11.H6', 'L11.H10', 'L9.H1', 'L11.H2', 
                    'L10.H10', 'L8.H7', 'L8.H6', 'L11.H8', 'L11.H7']
    low_layer_heads = ['L10.H8', 'L9.H2', 'L8.H10', 'L8.H4', 'L8.H5', 'L10.H6']

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'locations'):
        layers_heads = ['L10.H10', 'L11.H0', 'L11.H6']
    elif(property_type == 'objects'):
        layers_heads = ['L11.H7', 'L11.H8']
    elif(property_type == 'colors'):
        layers_heads = ['L11.H10']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(8, 12):
            for head in range(0, 12):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))
    
    return layers_heads

def get_vit_l_14_openai(property_type='high'):
    high_layer_heads = ['L21.H13', 'L22.H1', 'L22.H9', 'L21.H1', 'L21.H15', 'L21.H0', 'L20.H2', 'L20.H12', 'L22.H5', 
                        'L23.H4', 'L22.H15', 'L21.H8', 'L23.H10', 'L23.H11', 'L22.H2', 'L22.H13']
    low_layer_heads = ['L23.H5', 'L21.H3', 'L22.H14', 'L20.H15', 'L20.H6', 'L22.H6', 'L22.H4', 'L22.H0', 'L21.H2', 'L20.H4', 'L20.H1']
    layers = [20, 21, 22, 23]

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'animals'):
        layers_heads = ['L22.H13']
    elif(property_type == 'colors'):
        layers_heads = ['L23.H11']
    elif(property_type == 'locations'):
        layers_heads = ['L20.H2', 'L21.H0', 'L21.H1', 'L21.H13', 'L21.H15', 'L22.H2', 'L22.H5', 'L22.H15', 'L23.H10']
    elif(property_type == 'objects'):
        layers_heads = ['L22.H1', 'L23.H4']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(20, 24):
            for head in range(0, 16):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))
    elif(property_type == 'high-last'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-1]):
                layers_heads.append(each_layer_head)
    elif(property_type == 'high-secondlast'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-2]):
                layers_heads.append(each_layer_head)

    return layers_heads

def get_vit_l_14_laion(property_type='high'):
    high_layer_heads = ['L22.H1', 'L21.H1', 'L22.H5', 'L21.H9', 'L22.H3', 'L22.H6', 'L22.H12', 
                        'L22.H0', 'L23.H6', 'L23.H8', 'L21.H11', 'L23.H4', 'L22.H13', 'L21.H0', 
                        'L22.H10', 'L21.H5', 'L22.H8', 'L23.H5', 'L23.H9', 'L20.H14', 'L20.H4']
    low_layer_heads = ['L23.H1', 'L21.H6', 'L20.H13']
    layers = [20, 21, 22, 23]

    if(property_type == 'high'):
        layers_heads = high_layer_heads
    elif(property_type == 'low'):
        layers_heads = low_layer_heads
    elif(property_type == 'low-random'):
        all_layers_heads = low_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'high-random'):
        all_layers_heads = high_layer_heads
        layers_heads = random.sample(all_layers_heads, min(len(high_layer_heads), len(low_layer_heads)))
    elif(property_type == 'animals'):
        layers_heads = ['L22.H6']
    elif(property_type == 'colors'):
        layers_heads = ['L21.H0', 'L21.H9', 'L22.H10', 'L23.H8']
    elif(property_type == 'locations'):
        layers_heads = ['L21.H1', 'L21.H11', 'L22.H13', 'L23.H6']
    elif(property_type == 'objects'):
        layers_heads = ['L23.H3']
    elif(property_type == 'random'):
        all_layers_heads = []
        for layer in range(20, 24):
            for head in range(0, 16):
                cur_layer_head = 'L' + str(layer) + '.' + 'H' + str(head)
                if(cur_layer_head not in high_layer_heads):
                    all_layers_heads.append(cur_layer_head)

        layers_heads = random.sample(all_layers_heads, len(high_layer_heads))
    elif(property_type == 'high-last'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-1]):
                layers_heads.append(each_layer_head)
    elif(property_type == 'high-secondlast'):
        layers_heads = []
        for each_layer_head in high_layer_heads:
            if(int(each_layer_head.split('.')[0][1:]) == layers[-2]):
                layers_heads.append(each_layer_head)

    return layers_heads

def get_model_association(modelname, property_type='high'):

    if(modelname == 'ViT-B-32_openai'):
        return get_vit_b_32_openai(property_type)
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        return get_vit_b_32_datacomp(property_type)
    elif(modelname == 'ViT-B-16_openai'):
        return get_vit_b_16_openai(property_type)
    elif(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        return get_vit_b_16_laion(property_type)
    elif(modelname == 'ViT-L-14_openai'):
        return get_vit_l_14_openai(property_type)
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        return get_vit_l_14_laion(property_type)
