from tqdm import tqdm

import torch
import torch.nn.functional as F
import os
import clip
import datasets as dts


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, reduce='mean', gpt=False, wordnet_dict=None):
    with torch.no_grad():
        clip_weights = []
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:
                   
                    # Tokenize the prompts
                    name = name.replace('_', ' ')
                    
                    texts = [t.format(name) for t in template]
                    texts = clip.tokenize(texts).cuda()
        
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce=='mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i+=1
                indices.append(i)
                
            return clip_weights, indices
        else:
        
            for classname in classnames:
                
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                
                if gpt:
                    texts = template[classname]
                else:
                    texts = [t.format(classname)  for t in template]
                texts = clip.tokenize(texts).cuda()
    
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if reduce=='mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)
        
            clip_weights = torch.stack(clip_weights, dim=-1).cuda()
    return clip_weights


def get_all_features(args, test_loader, dataset, clip_model, backbone_name):
    clip_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model, reduce=None)
    test_features, test_labels = pre_load_features(args, "test", clip_model, test_loader, backbone_name=backbone_name)

    return test_features, test_labels, clip_prototypes


def build_cache_model(cfg, clip_model, train_loader_cache, n_views=0, reduce=None):
    print('... for shot samples from train split:')

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        if n_views == 0:
            n_epochs =1
        else:
            n_epochs = n_views
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(n_epochs):
                train_features = []
                train_labels = []
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                        
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))


        
        if n_views == 1:
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            #cache_keys = cache_keys.permute(1, 0)
        else:
            cache_keys = torch.cat(cache_keys, dim=0) # [n_views, n_classes, n_features]
            if reduce == 'mean':
                cache_keys = cache_keys.mean(0, keepdim=True)
                
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys.permute(0, 2, 1)
            
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values

def pre_load_features(args, split, clip_model, loader, n_views=1, backbone_name = 'ViT-B/16',):
    model_cache_name = backbone_name.replace('/', '_')
    model_cache_name = model_cache_name.replace('-', '_')

    if  not args.load:
        features, labels = [], []
        
        with torch.no_grad():
          
            for view in range(n_views):
                length = 0
                for i, (images, target) in enumerate(tqdm(loader)):
                    if n_views == 1:
                        
                        images, target = images.cuda(), target.cuda()
                        
                        
                        image_features = clip_model.encode_image(images)
                        
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        
                        
                        features.append(image_features.cpu())
                        labels.append(target.cpu())
                    else:
                        images, target = images.cuda(), target.cuda()
                        image_features = clip_model.encode_image(images)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        if view == 0:
                            labels.append(target.cpu())
                            if i ==0:
                                mean_features = image_features
                            else:
                                mean_features = torch.cat((mean_features, image_features))
                        else:
                            mean_features[length:length+image_features.size(0)] += image_features
                            length += image_features.size(0)
                            
        if n_views > 1:
            mean_features = mean_features / n_views
            features = mean_features / mean_features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels)
        
        elif n_views==1:
            features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, args.cache_dir + "/" +  f'{model_cache_name}_' + split + "_features.pt")
        torch.save(labels, args.cache_dir + "/" + split + "_target.pt")
        
    else:
        try:
            features = torch.load(args.cache_dir + "/" +   f'{model_cache_name}_' + split + "_features.pt")
            labels = torch.load(args.cache_dir + "/" + split + "_target.pt")
        except FileNotFoundError:
            print("Cache not found...")
    
    return features, labels


def get_samples_feature_and_labels(cache_dir, splits = ['test'], backbone_name = 'ViT-B/16', dataset_name = ''):
    if 'EVA' not in backbone_name:
        model_cache_name = backbone_name.replace('/', '_')
        model_cache_name = model_cache_name.replace('-', '_')
        out = []
            
        for spl in splits:
            features_path = os.path.join(cache_dir, f'{model_cache_name}_{spl}_features.pt')
            try:
                _features = torch.load(features_path).cuda()
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not find cached features at {features_path}. Run the command without --load first. ')
            _labels = torch.load(os.path.join(cache_dir, f'{spl}_target.pt')).cuda()
            out.append(_features)
            out.append(_labels)
    else:
        out = []
        assert splits == ['test']
        root_eva_features = '/export/DATA/mzanella/datasets/EVA_features/'
        eva_dnames_dic = {
        'sun397':'sun397',
        'imagenet':'imagenet',
        'fgvc_aircraft':'fgvc',
        'eurosat':'eurosat',
        'food101':'food101',
        'caltech101':'caltech101',
        'oxford_pets':'oxford_pets',
        'oxford_flowers':'oxford_flowers',
        'stanford_cars':'stanford_cars',
        'dtd':'dtd',
        'ucf101':'ucf101',
         }
        eva_features_path = os.path.join(root_eva_features, eva_dnames_dic[dataset_name], 'BAAI', backbone_name,)
        _features = torch.load(os.path.join(eva_features_path, 'image_features.pt')).cuda()
        out.append(_features)
        
        for spl in splits:
            _labels = torch.load(os.path.join(cache_dir, f'{spl}_target.pt')).cuda()
            out.append(_labels)
    
    return out


def load_features(dataset_name, 
                  root_path, 
                  cache_dir, 
                  preprocess, 
                  clip_model, 
                  backbone_name,
                  splits = ['train', 'test'],
                  load_loaders = False,):

    cfg = {}
    #print(f'============ DATASET : {dataset_name}')
    
    
    cfg['dataset'] = dataset_name #datasets[dataset_name]
    
    cfg['root_path'] = root_path 
    cfg['shots'] = 0
    cfg['load_pre_feat'] = True
    cfg['cache_dir'] = cache_dir
    if dataset_name == 'imagenet':
        cfg['load_cache'] = False
    if load_loaders:
        train_loader, val_loader, test_loader, dataset = dts.get_all_dataloaders(cfg, preprocess, dirichlet=None)
    else:
        if dataset_name != 'imagenet':
            dataset = dts.dataset_list[dataset_name](cfg['root_path'], cfg['shots'])
        else:
            dataset = dts.dataset_list[dataset_name](cfg['root_path'], cfg['shots'], None)
        train_loader, val_loader, test_loader = None,None,None

    features_and_labels = get_samples_feature_and_labels(cache_dir,
                                                        splits = splits,
                                                        backbone_name = backbone_name,
                                                        dataset_name = dataset_name,
                                                        )

    return train_loader, val_loader, test_loader, dataset, features_and_labels


import nltk
from nltk.corpus import wordnet
import pickle

def get_wordnet_prompts(args):
    # Set custom NLTK data directory
    nltk_data_path = os.path.join(args.root_path, 'wordnet_data')
    os.makedirs(nltk_data_path, exist_ok=True)  # Ensure the directory exists
    nltk.data.path.append(nltk_data_path)

    # Download WordNet to the custom path
    nltk.download('wordnet', download_dir=nltk_data_path)

    # Verify that NLTK is using the custom path

    # Get all words from WordNet
    all_words = set()
    root_terms = [
                  'building', 
                  'vehicle', 
                  'food', 
                  'flower', 
                  'animal', 
                  'texture', 
                  'action',
                  'furniture'
                  ]
    for r in root_terms:
        #for synset in tqdm(wordnet.all_synsets(pos = 'n')):
        for synset in wordnet.synsets(r, pos='n'):
            li_words = synset.lemma_names()
            cleaned_li_words = []
            for w in li_words:
                if not(w[0] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    cleaned_li_words.append(w.replace('_', ' ').replace('.', ''))

            all_words.update(cleaned_li_words)
            #print(cleaned_li_words)
            for hyponym in synset.closure(lambda s: s.hyponyms()):
                li_words = [w.replace('_', ' ').replace('.', '') for w in hyponym.lemma_names()]
                all_words.update(li_words)
                #print(li_words)

    # Convert to a sorted list
    all_words = sorted(all_words)

    # # Print some sample words
    # print(all_words[117:133])  # Print first 10 words as an example
    # print('total number of words', len(all_words))  # Print first 10 words as an example
    
    with open(os.path.join(args.root_path, 'filtered_wordnet_words.pickle'), 'wb') as f:
        pickle.dump(all_words, f)
        
    backbones = {
                 'vit_b16': 'ViT-B/16',
                 'rn50': 'RN50',
                 'vit_b32':'ViT-B/32',
                 'rn101': 'RN101',    
                 'vit_l14':'ViT-L/14'   
                }

    model_name = args.backbone
    
    try:
        with open(os.path.join(args.root_path, f'filtered_wordnet_textual_features_{model_name}.pickle'), 'rb') as f:
            all_textual_features = pickle.load(f)
    except FileNotFoundError:
        all_textual_features = None 
        clip_model, preprocess = clip.load(backbones[model_name])
        batch_size = 64
        current_start = 0
        current_end = 0
        texts = []
        for jw , w in enumerate(tqdm(all_words)):
            texts.append(f'A photo of a {w}.')
            
            if len(texts) == batch_size or jw == len(all_words):
                with torch.no_grad(), torch.amp.autocast('cuda', dtype = torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    current_end = current_start + len(texts)
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if all_textual_features is None:
                        all_textual_features = torch.zeros((len(all_words),class_embeddings.shape[-1]), dtype = torch.float16)
                    all_textual_features[current_start:current_end,:] = class_embeddings[...].cpu()
                    current_start = current_end
                    texts = []
        with open(os.path.join(args.root_path, f'filtered_wordnet_textual_features_{model_name}.pickle'), 'wb') as f:
            pickle.dump(all_textual_features, f)

    return all_textual_features.cuda().T