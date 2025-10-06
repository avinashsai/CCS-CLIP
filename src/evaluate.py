import os
import pickle as pkl
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch

from prompts import (
    get_templates
)
from labels import (
    get_occupations
)
from visual_utils import (
    read_frames_decord, 
    read_image, 
    VideoCapture
)
from metrics import (
    skew_metric,
    ret_metrics
)

def get_visual_features(visual_input, preprocess, model, num_frames, input_res=224):
        custom_transforms = transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        pixel_values = custom_transforms(visual_input)
        pixel_values = pixel_values.reshape(-1, 3, input_res, input_res)
        visual_features = model.encode_image(pixel_values)
        visual_features = visual_features.reshape(1, num_frames, -1)
        visual_features /= visual_features.norm(dim=-1, keepdim=True)
        visual_features = torch.mean(visual_features, 1)
        return visual_features

def get_text_features(texts, tokenizer, model, device='cuda'):
    texts = tokenizer(texts).to(device) #tokenize
    class_embeddings = model.encode_text(texts) #embed with text encoder
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    #print(class_embeddings.size())
    return class_embeddings

def evaluate_image(test_images, test_labels, text_weights, preprocess, model, device='cuda'):
    model.eval()
    with torch.no_grad():
        top1, n = 0., 0.
        for ii in tqdm(range(len(test_images))):
            pil_image = Image.open(test_images[ii])
            images = preprocess(pil_image)[np.newaxis, :, :, :]
            target = test_labels[ii]

            images = images.to(device)
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ text_weights

            #print(logits.size())

            pred = torch.argmax(logits).item()

            acc1 = (pred == target)
            top1 += acc1
            n += images.size(0)

    top1 = (top1 / n) * 100
    return top1

def get_video_features(model, test_sentences, preprocess, video_path, num_frames,
                    taskname=None,
                    sample_type='uniform',
                    device='cuda',
                    is_custom_transforms=False):
    
    model.eval()
    with torch.no_grad():
        visual_id_features_dict = {}
        for test_sen_index in tqdm(range(len(test_sentences))):
            try:
                if(isinstance(test_sentences[test_sen_index], tuple) == True):
                    video, _ = test_sentences[test_sen_index]
                else:
                    video = test_sentences[test_sen_index]
                #print(video, flush=True)
                if(video not in visual_id_features_dict):
                    if((video.lower().endswith('jpeg')) or (video.lower().endswith('png')) or (video.lower().endswith('jpg'))):
                        imgs = read_image(os.path.join(video_path, video))
                    elif((video.lower().endswith('gif'))):
                        imgs, _ = VideoCapture.load_frames_from_video(os.path.join(video_path, video), num_frames, 
                                                        sample_type)
                    else:
                        #print(video, flush=True)
                        imgs, _ = read_frames_decord(os.path.join(video_path, video), num_frames, 
                                                    sample_type)
                        #if('kinetics' not in taskname):
                        #    imgs, _ = read_frames_decord(os.path.join(video_path, video), num_frames, 
                        #                            sample_type)
                        #else:
                        #    imgs, _ = VideoCapture.load_frames_from_video(os.path.join(video_path, video), num_frames, 
                        #                                sample_type)

                    imgs = imgs.to(device)
                    visual_out = get_visual_features(imgs, preprocess, model, num_frames)

                    if(isinstance(visual_out, tuple) == True):
                        visual_id_features_dict[video] = [visual_out[0].detach().cpu().numpy(), visual_out[1].detach().cpu().numpy()]
                    else:
                        visual_id_features_dict[video] = visual_out.detach().cpu().numpy()
            except:
                continue

    return visual_id_features_dict

def evaluate_video_ret(model, test_sentences, test_labels, 
                               preprocess,
                               tokenizer,
                               video_path, 
                               num_frames,
                               taskname,
                               modelname,
                               batch_size,
                               sample_type='uniform', 
                               device='cuda',
                               retmode='t2v'):

    visual_id_features_dict = get_video_features(model=model, test_sentences=test_sentences,
                                                preprocess=preprocess,
                                                video_path=video_path,
                                                num_frames=num_frames,
                                                taskname=taskname,
                                                sample_type=sample_type,
                                                device=device,
                                                )

    print(f'\nTotal unique videos: {len(visual_id_features_dict)}')

    visual_features = []
    text_features = []

    num_videos = len(test_sentences)
    model.eval()
    with torch.no_grad():
        if(num_videos % batch_size == 0):
            num_batches = int(num_videos // batch_size)
        else:
            num_batches = int(num_videos // batch_size) + 1
        
        for batch_id in range(num_batches):
            start = batch_id * batch_size
            end = min((batch_id + 1) * batch_size, num_videos)
            text = test_labels[start : end]

            text_out = get_text_features(text, tokenizer, model).detach().cpu().numpy()
            text_features += text_out.tolist()

            visual_out = [visual_id_features_dict[video] for video in test_sentences[start : end]]
            visual_features += visual_out

        text_features = np.asarray(text_features).reshape(num_videos, -1)
        visual_features = np.asarray(visual_features).reshape(num_videos, -1)

        print("*************************************************************************************")
        print(f'\nText features: {text_features.shape}')
        print(f'\nVideo features: {visual_features.shape}')        
    return ret_metrics(text_features, visual_features, retmode)

def evaluate_fairness_occupation(test_images, dataset, attribute, modelname, property_type,
                                model, preprocess, tokenizer, topk, device='cuda'):

    text_templates = get_templates(dataset)
    occupations = get_occupations(dataset)

    max_skew = 0.0
    min_skew = 0.0
    model.eval()
    with torch.no_grad():
        if(dataset == 'socialcounterfactuals-skew'):
            fname = dataset + '-' + attribute + '-' + property_type + '-' + modelname + '.pkl'
            with open('image_fea/' + fname, 'rb') as f:
                all_similarities = pkl.load(f)
            for ii in tqdm(range(len(occupations))):
                each_occupation = occupations[ii]
                text_prompts = [sentence.replace('<occupation>', each_occupation) for sentence in text_templates]
                #print(text_prompts)
                text_weights = get_text_features(texts=text_prompts, tokenizer=tokenizer, model=model, device=device)
                text_weights = text_weights.mean(dim=0)
                text_weights /= text_weights.norm()

                similarity_scores = all_similarities[ii]
                counts = []
                for jj in range(len(test_images[ii])):
                    counts.append(len(test_images[ii][jj]))
                    #for kk in range(len(test_images[ii][jj])):
                    #    #print(test_images[ii][jj][kk])
                    #    pil_image = Image.open(test_images[ii][jj][kk])
                    #    images = preprocess(pil_image)[np.newaxis, :, :, :]
                    #    images = images.to(device)
                        
                        # predict
                    #    image_features = model.encode_image(images)
                    #    image_features /= image_features.norm(dim=-1, keepdim=True)
                    #    cosine_score = (text_weights @ image_features.T).item()
                    #    similarity_scores.append((cosine_score, jj))
                #all_similarities.append(similarity_scores)

                total_cnt = sum(counts)
                similarity_scores.sort(key=lambda x: x[0], reverse=True)
                skewness = skew_metric(similarity_scores, counts, total_cnt, topk)
                skewness = np.asarray(skewness)

                max_skew += np.max(skewness)
                min_skew += np.min(skewness)

            #fname = dataset + '-' + attribute + '-' + property_type + '-' + modelname + '.pkl'
            #with open('image_fea/' + fname, 'wb') as f:
            #    pkl.dump(all_similarities, f)
            #with open('image_fea/' + fname, 'rb') as f:
            #    image_rep = pkl.load(f)
        else:
            counts_dict = {}
            for ii in tqdm(range(len(test_images))):
                #pil_image = Image.open(test_images[ii][0])
                #images = preprocess(pil_image)[np.newaxis, :, :, :]
                #images = images.to(device)
                
                # predict
                #image_features = model.encode_image(images)
                #image_features /= image_features.norm(dim=-1, keepdim=True)
                #image_features = image_features.detach().cpu()
                #image_rep.append((image_features, test_images[ii][2]))

                if(test_images[ii][1] in counts_dict):
                    counts_dict[test_images[ii][1]] += 1
                else:
                    counts_dict[test_images[ii][1]] = 1
            
            counts = [counts_dict[val] for val in counts_dict.keys()]
            total_cnt = sum(counts)
            print(counts_dict)
            print(counts)

            fname = dataset + '-' + attribute + '-' + property_type + '-' + modelname + '.pkl'
            #with open('image_fea/' + fname, 'wb') as f:
            #    pkl.dump(image_rep, f)
            with open('image_fea/' + fname, 'rb') as f:
                image_rep = pkl.load(f)
            
            for ii in tqdm(range(len(occupations))):
                each_occupation = occupations[ii]
                text_prompts = [sentence.replace('<occupation>', each_occupation) for sentence in text_templates]
                #print(text_prompts)
                text_weights = get_text_features(texts=text_prompts, tokenizer=tokenizer, model=model, device=device)
                text_weights = text_weights.mean(dim=0)
                text_weights /= text_weights.norm()
                text_weights = text_weights.detach().cpu()

                similarity_scores = [((text_weights @ image_features[0].T).item(), image_features[1]) for image_features in image_rep]
                
                similarity_scores.sort(key=lambda x: x[0], reverse=True)
                skewness = skew_metric(similarity_scores, counts, total_cnt, topk)
                skewness = np.asarray(skewness)

                max_skew += np.max(skewness)
                min_skew += np.min(skewness)
    
    return (max_skew, min_skew)
