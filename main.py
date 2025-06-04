import os
import random
import argparse
import numpy as np
import torch
from datasets import get_all_dataloaders
import utils as uti
from tqdm import tqdm
import clip
import datasets as dts

def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument('--dataset', default='dtd', help='dataset name', type=str)
    parser.add_argument('--root_path', default='./datasets', type=str)
    parser.add_argument('--source_prompts_types', 
                        default='imagenet_text', type=str, choices=['imagenet_text', 
                                                                    'imagenet_images', 
                                                                    'wordnet', ])
    parser.add_argument('--method', default = 'SiM', type = str, choices = ['SiM'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', 
                        type=str, 
                        choices=['rn50', 'rn101', 'vit_b32', 'vit_b16', 'vit_l14'], 
                        help="CLIP architecture")
    # parser.add_argument('--cache_dir', type = str, default = None, help='Where to store visual and textual features if not None')
    parser.add_argument('--load', action='store_true', default=False, help="Load features from cache_dir")
    parser.add_argument('--n_shots', type=int, default = 16)

    # Experimental arguments
    parser.add_argument('--n_random_seeds', type=int, default=10, help="Number of random seeds")

    args = parser.parse_args()
    return args




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_shots(args, base_rng, train_labels, train_features):
    shots_seed = base_rng.integers(0,10000)
    #print('shots_seed: ', shots_seed)
    shots_rng = np.random.default_rng(shots_seed)
    n_shots = args.n_shots
    K = torch.max(train_labels)+1
    all_train_indexes = torch.tensor(range(train_features.shape[0]))
    shots_features = torch.zeros((n_shots*K, *train_features.shape[1:]), dtype = train_features.dtype)
    shots_labels = torch.zeros((n_shots*K), dtype = train_labels.dtype)
    shots_indexes = torch.zeros((n_shots*K), dtype = torch.int64)
    for k in range(K):
        mask = train_labels==k
        selected_shots_idx = shots_rng.choice(torch.sum(mask).cpu(), n_shots, replace = False)
        shots_indexes[k*n_shots:(k+1)*n_shots] = all_train_indexes[mask.cpu()][torch.tensor(selected_shots_idx)]
        shots_features[k*n_shots:(k+1)*n_shots,...] = train_features[mask,...][selected_shots_idx,...]
        shots_labels[k*n_shots:(k+1)*n_shots] = k
    return shots_features, shots_labels, shots_indexes

def load_source_prototypes(args, clip_model):
    if args.source_prompts_types == 'imagenet_text':
        clip_prototypes = uti.clip_classifier(dts.imagenet.imagenet_classes, 
                                          dts.imagenet.imagenet_templates, 
                                          clip_model,
                                          reduce = 'mean')
        
    elif args.source_prompts_types == 'wordnet':
        raise RuntimeError('TODO: recompute prototypes instead of loading pickles')
    elif args.source_prompts_types == 'imagenet_images':
        raise RuntimeError('TODO')
    return clip_prototypes

def main():

    args = get_arguments()
    
    set_random_seed(args.seed) # for reproducibility
    base_rng = np.random.default_rng(args.seed)
    
    

    # CLIP model
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}
    clip_model, preprocess = clip.load(backbones[args.backbone])
    clip_model.eval()

    # Prepare dataset
    datasets = {
        'imagenet':'imagenet',
        'sun397':'sun397',
        'fgvc_aircraft':'fgvc_aircraft',
        'eurosat':'eurosat',
        'stanford_cars':'StanfordCars',
        'food101':'Food101',
        'oxford_pets':'OxfordPets',
        'oxford_flowers':'Flower102',
        'caltech101':'Caltech101',
        'dtd':'dtd',
        'ucf101':'UCF101',
                 }
    assert args.dataset in datasets.keys(), print(f'Could not find {args.dataset} in possible datasets {list(datasets.keys())}')
    train_loader, val_loader, test_loader, dataset = get_all_dataloaders(args, preprocess)
    #print(dataset.classnames)
    # Load features
    args.cache_dir = os.path.join(args.root_path, datasets[args.dataset], 'cache') # always use automatic cache dir to avoid mixups
    os.makedirs(args.cache_dir, exist_ok=True)
    if not args.load:
        _ = uti.pre_load_features(args, 'test', clip_model, test_loader, backbone_name = backbones[args.backbone])
        _ = uti.pre_load_features(args, 'train', clip_model, train_loader, backbone_name = backbones[args.backbone])
        args.load = True
    if args.load:
            
        
        train_loader, val_loader, test_loader, dataset,\
        features_and_labels\
        = uti.load_features(args.dataset, 
                            args.root_path, 
                            args.cache_dir, 
                            preprocess, 
                            clip_model,
                            backbones[args.backbone],
                            splits = ['train','test'],
                            load_loaders=False)
        train_features, train_labels, test_features, test_labels = features_and_labels
    # load source prototypes
    source_prototypes = load_source_prototypes(args, clip_model)
    clip_model = clip_model.to('cpu')  # unload CLIP model from VRAM
    
    # get lambda reg values
    if args.source_prompts_types == 'imagenet_text':
        lambda_reg = 0.1
    
    # select shots
    mapped_accs = torch.zeros(args.n_random_seeds)
    for jseed in tqdm(range(args.n_random_seeds)):
        shots_features, shots_labels, shots_indexes = select_shots(args, base_rng, train_labels, train_features)
        
        
        # solve with Tikhonov regularized least square
        L_P = (shots_features.cuda()@source_prototypes.squeeze()).cpu().float()
        Y = torch.nn.functional.one_hot(shots_labels)
        
        
        I_kp = torch.eye(L_P.shape[-1])
        W = torch.linalg.solve(L_P.T @ L_P + lambda_reg * I_kp , L_P.T @ Y.float())
        
        W = W.half().cuda()
        test_source_logits = 100.*test_features.cuda()@source_prototypes.squeeze()
        mapped_logits = test_source_logits@W 
        mapped_pred = torch.argmax(mapped_logits, dim = -1).cuda()
        mapped_acc = torch.sum(mapped_pred.cpu() == test_labels.cpu())/mapped_pred.shape[0]
        mapped_accs[jseed] = mapped_acc
        
    acc_tot = mapped_accs.mean()
     
    print("\n============================")
    print("      Final Results         ")
    print("============================")
    print(f"Dataset:         {args.dataset}")
    print(f"Backbone:        {args.backbone}")
    print(f"Number of Random Seeds: {args.n_random_seeds}")
    print(f"Method: {args.method}")
      
      
    print("----------------------------")
    print(f"FINAL Accuracy:     {acc_tot:.4f}")
    print("============================\n")



if __name__ == '__main__':
    main()
