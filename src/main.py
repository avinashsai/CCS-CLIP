import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from prompts import get_templates
from load_model import get_model_tokenizer, modify
from load_image_data import (
    get_images, 
    get_label_names,
    get_fairness_dataset
)
from labels import (
    get_occupations,
    get_data_paths
)
from load_video_data import (
    load_video_ret,
    get_data_paths
)
from zero_shot_utils import zeroshot_classifier, get_accuracy
from evaluate import (
    evaluate_image,
    evaluate_video_ret,
    evaluate_fairness_occupation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_classification_datasets = ['cifar10', 'cifar10animals', 'cifar10objects', 'cifar100', 'food101',
                                'country211', 'inaturalist', 'fmnist', 'oxfordpets', 'imagenet-a', 'imagenet-r',
                                'fairface-class', 'socialcounterfactuals-class']
video_retrieval_datasets = ['msrvtt-ret', 'msvd-ret', 'didemo-ret']
fairness_datasets = ['fairface-skew', 'socialcounterfactuals-skew']

all_models = {}
all_models['high'] = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
            'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']
all_models['low'] = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
            'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']

all_models['animals'] = ['ViT-B-16_openai', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']
all_models['objects'] = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k',
            'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']
all_models['locations'] = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
            'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']

all_models['random'] = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
            'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']
all_models['colors'] = all_models['high']
all_models['high-last'] = all_models['high']
all_models['high-secondlast'] = all_models['high']
all_models['all'] = all_models['high']
all_models['low-random'] = all_models['high']
all_models['high-random'] = all_models['high']

frames_dict = {'msvd-qa': 12,
               'msrvtt-qa': 12,
               'imagenet1k': 1,
               'msrvtt-ret': 12,
               'msvd-ret': 12,
               'didemo-ret': 32,
               'ucf-act-rec': 1,
               'hmdb51-act-rec': 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='all', 
                        choices=['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
                                'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k',
                                'all'], 
                        help='Type of tasks')
    parser.add_argument('--dataset', type=str, default='fairface', help='Dataset')
    parser.add_argument('--property_type', type=str, default='all', help='Type of property')
    parser.add_argument('--attribute', type=str, default=None, help='Atrribute')
    parser.add_argument('--topk', type=int, default=10, help='Number of retrieved images')

    is_joined = False
    is_associated = False
    runs = 1

    args = parser.parse_args()
    modelname = args.modelname
    dataset = args.dataset
    property_type = args.property_type
    attribute = args.attribute
    topk = int(args.topk)

    if(property_type != 'all'):
        is_associated = True
    if(property_type.endswith('random') is True):
        runs = 3

    if(dataset in ['cifar10', 'cifar10animals', 'cifar10objects', 'cifar100', 'food101']):
        is_joined = True

    avg_acc = 0.0

    if(dataset in ['fairface-skew']):
        modelnames = ['ViT-B-32_openai', 'ViT-B-32_datacomp_m_s128m_b4k', 'ViT-B-16_openai',
                    'ViT-B-16_laion2b_s34b_b88k', 'ViT-L-14_openai', 'ViT-L-14_laion2b_s32b_b82k']
    elif(dataset in ['socialcounterfactuals-skew']):
        modelnames = ['ViT-B-32_openai', 'ViT-B-16_openai', 'ViT-L-14_openai']
    elif(modelname == 'all'):
        modelnames = all_models[property_type]
    else:
        modelnames = [modelname]

    skewness_scores = []
    for modelname in modelnames:
        print("###########################################################################################")
        print(f'Evaluating on: {dataset}, Model name: {modelname}, Property-type: {property_type}')
        print(f'Attribute: {attribute}')
        if(is_associated):
            print(f'Property evaluated: {property_type}')

        model, tokenizer, preprocess = get_model_tokenizer(modelname, device)
        #for name, layer in model.named_modules():
        #    print(f"{name}: {layer}")
            
        model = modify(model, modelname, is_associated, property_type)
        model.to(device)
        model.eval()

        print("###########################################################################################")
        if(dataset in image_classification_datasets):
            if(dataset in ['fairface-class', 'socialcounterfactuals-class']):
                test_images, test_labels = get_fairness_dataset(dataset, attribute, is_joined)
            else:
                test_images, test_labels = get_images(dataset, is_joined)

            print(test_images[0:5])
            print(test_labels[0:5])
            print(f'Test size images: {len(test_images)}')
            print(f'Test size labels: {len(test_labels)}')

            dataset_classnames = get_label_names(dataset, attribute)
            dataset_templates = get_templates(dataset)

            print(f'Dataset class names: {dataset_classnames}')

            text_weights = zeroshot_classifier(model, tokenizer, dataset_classnames, dataset_templates, device)
            print(f'Text prompts size: {text_weights.size()}')

            top1 = evaluate_image(test_images, test_labels, text_weights, preprocess, model, device)
            print(f'Dataset: {dataset}, Model: {modelname}, zero-shot-accuracy: {top1:.2f}')
            avg_acc += top1
        elif(dataset in video_retrieval_datasets):
            test_sentences, test_labels = load_video_ret(dataset)
            batchsize = 16

            print(test_sentences[0:5])
            print(test_labels[0:5])

            print(len(test_sentences))
            print(len(test_labels))

            scores = evaluate_video_ret(model=model, test_sentences=test_sentences,
                                test_labels=test_labels,
                                preprocess=preprocess,
                                tokenizer=tokenizer,
                                video_path=get_data_paths(dataset),
                                num_frames=frames_dict[dataset],
                                taskname=dataset,
                                modelname=modelname,
                                batch_size=batchsize,
                                retmode='t2v')

            print(f'Dataset: {dataset}, Model: {modelname}, retrieval-scores: {scores}')
        elif(dataset in fairness_datasets):
            test_images = get_fairness_dataset(dataset, attribute)

            if(dataset == 'socialcounterfactuals-skew'):
                print(test_images[0][0][0:5])
                print(test_images[0][1][0:5])
                print(test_images[1][0][0:5])
                print(test_images[1][1][0:5])
            else:
                print(test_images[0:5])
            
            scores = evaluate_fairness_occupation(test_images=test_images, 
                                    dataset=dataset,
                                    attribute=attribute,
                                    modelname=modelname,
                                    property_type=property_type,
                                    model=model, 
                                    preprocess=preprocess, 
                                    tokenizer=tokenizer, 
                                    topk=topk, 
                                    device='cuda')
            
            skewness_scores.append((modelname, scores[0]))
            print(f'Dataset: {dataset}, Atrribute: {attribute}, Model: {modelname}')
            print(f'Propert type: {property_type}')
            print(f'Max-skew: {scores[0]:.2f}, Min-skew: {scores[1]:.2f}')

        print("###########################################################################################")
        if(property_type == 'random'):
            print(f'Dataset: {dataset}, Model: {modelname}, average-zero-shot-accuracy: {(avg_acc / runs):.2f}')
    if(dataset in fairness_datasets):
        print(f'Topk: {topk}')
        for x in skewness_scores:
            print(f'{x[0]}: {x[1]:.2f}')

if __name__ == '__main__':
    main()
