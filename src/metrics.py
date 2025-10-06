import numpy as np
from tqdm import tqdm
import torch
#from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap

def skew_metric(predictions, counts, total_cnt, topk):
    results = [0] * len(counts)
    #print(predictions)
    #print(topk)

    for ii in range(topk):
        #print(int(predictions[ii][1]), end=" ", flush=True)
        results[int(predictions[ii][1])] += 1

    #print(counts)
    #print(results)

    skewness = []
    for ii in range(len(results)):
        if(results[ii]):
            skewness.append(np.log((results[ii] / topk) / (counts[ii] / total_cnt)))
    
    return skewness

def ret_metrics(text_features, visual_features, retmode='t2v'):
    text_size, _ = text_features.shape
    visual_size, _ = visual_features.shape

    pred_rankings = []
    if(retmode == 't2v'):
        for ii in tqdm(range(text_size)):
            scores = [-(text_features[ii] @ visual_features[jj, :]) for jj in range(visual_size)]
            pred_rankings.append(np.argwhere(np.argsort(scores) == ii)[0][0])
        total_cnt = text_size
    else:
        for ii in tqdm(range(visual_size)):
            scores = [-(visual_features[ii] @ text_features[jj, :]) for jj in range(text_size)]
            pred_rankings.append(np.argwhere(np.argsort(scores) == ii)[0][0])
        total_cnt = visual_size

    pred_rankings = np.asarray(pred_rankings)
    
    metrics = {}
    metrics["R1"] = round(100 * float(np.sum(pred_rankings == 0)) / total_cnt, 1)
    metrics["R5"] = round(100 * float(np.sum(pred_rankings < 5)) / total_cnt, 1)
    metrics["R10"] = round(100 * float(np.sum(pred_rankings < 10)) / total_cnt, 1)
    metrics["R50"] = round(100 * float(np.sum(pred_rankings < 50)) / total_cnt, 1)
    metrics["MedR"] = round(np.median(pred_rankings) + 1, 1)
    metrics["MeanR"] = round(np.mean(pred_rankings) + 1, 1)

    return metrics

'''
def cap_metrics(ref_file, hypo_file):
    # create coco object and cocoRes object
    coco = COCO(ref_file)
    cocoRes = coco.loadRes(hypo_file)
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)
    # evaluate on a subset of images by setting
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    cocoEval.evaluate()
    # print output evaluation scores
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
        print('%s: %.3f'%(metric, score))
    #for key,value in cocoEval.imgToEval.items():
        #print(key,value)
    return results
'''
