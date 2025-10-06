import numpy as np
import cv2
import torch
import clip
import open_clip
from torch.nn import functional as F
from transformers import (
    CLIPTokenizer
)
import einops
from utils.factory import create_model_and_transforms, get_tokenizer
from association import (
    get_model_association,
)


def get_model_tokenizer(modelname, device):
    if(modelname == 'ViT-B-16_laion2b_s34b_b88k'):
        model_name = 'ViT-B-16'
        pretrained = 'laion2b_s34b_b88k'
    elif(modelname == 'ViT-B-16_openai'):
        model_name = 'ViT-B/16'
        pretrained = 'openai'
        tokenizer_name = 'ViT-B-16'
    elif(modelname == 'ViT-B-32_datacomp_m_s128m_b4k'):
        model_name = 'ViT-B-32'
        pretrained = 'datacomp_m_s128m_b4k'
    elif(modelname == 'ViT-B-32_openai'):
        model_name = 'ViT-B/32'
        pretrained = 'openai'
        tokenizer_name = 'ViT-B-32'
    elif(modelname == 'ViT-L-14_laion2b_s32b_b82k'):
        model_name = 'ViT-L-14'
        pretrained = 'laion2b_s32b_b82k'
    elif(modelname == 'ViT-L-14_openai'):
        model_name = 'ViT-L/14'
        pretrained = 'openai'
        tokenizer_name = 'ViT-L-14'

    if(pretrained == 'openai'):
        model, preprocess = clip.load(model_name, device=device)
        tokenizer = get_tokenizer(tokenizer_name)
        #tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)

    #model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    #model.to(device)
    #model.eval()

    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #tokenizer = get_tokenizer(model_name)

    return model, tokenizer, preprocess

def modify(model, modelname, is_associated=False, property_type='high'):
    if(not is_associated):
        return model
    
    model_heads = get_model_association(modelname, property_type)
    print(model_heads)

    vit_model = model.visual.transformer

    for each_layer_head in model_heads:
        layer_idx = int(each_layer_head.split('.')[0][1:])
        head_idx = int(each_layer_head.split('.')[1][1:])

        print((layer_idx, head_idx))

        # Get the multi-head attention block in the specific layer
        attn_layer = vit_model.resblocks[layer_idx].attn

        num_heads = attn_layer.num_heads  # Number of attention heads
        embed_dim = attn_layer.embed_dim  # Embedding dimension
        head_dim = embed_dim // num_heads  # Dimension per head

        # Zero-out specific head in Q, K, and V projections
        with torch.no_grad():
            attn_layer.in_proj_weight[head_idx * head_dim : (head_idx + 1) * head_dim, :] = 0
            attn_layer.in_proj_bias[head_idx * head_dim : (head_idx + 1) * head_dim] = 0
            attn_layer.out_proj.weight[:, head_idx * head_dim : (head_idx + 1) * head_dim] = 0

    return model
