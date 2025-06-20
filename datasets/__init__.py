import torch
import torchvision.transforms as transforms
from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .imagenet_a import ImageNetA
from .imagenet_v2 import ImageNetV2
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .utils import *

dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "fgvc_aircraft": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "imagenet_a": ImageNetA,
                "imagenet_v2": ImageNetV2,
                "imagenet_r": ImageNetR,
                "imagenet_sketch": ImageNetSketch,
                }


def get_all_dataloaders(args, preprocess, batch_size = 64, num_workers = 8):
    dataset_name = args.dataset

    if dataset_name.startswith('imagenet'):
        sampler = None
 
        dataset = dataset_list[dataset_name](args.root_path, cfg['shots'], 
                                             preprocess=preprocess, 
                                             train_preprocess=None, 
                                             test_preprocess=None, 
                                             load_cache=cfg['load_cache'], 
                                             load_pre_feat=cfg['load_pre_feat'])
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=batch_size, num_workers=num_workers, shuffle=False, sampler=sampler)
        train_loader = None
        val_loader = None
        if cfg['shots'] > 0:
            train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            
    else:
        dataset = dataset_list[dataset_name](args.root_path, 0)
        val_loader = build_data_loader(data_source=dataset.val, batch_size=batch_size, is_train=False, tfm=preprocess,
                                       shuffle=False)

        sampler = None

        test_loader = build_data_loader(data_source=dataset.test, batch_size=batch_size, is_train=False, tfm=preprocess,
                                        shuffle=False, sampler=sampler, num_workers = num_workers)


        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=batch_size, tfm=preprocess,
                                               is_train=False, shuffle=False, num_workers = num_workers)

    return train_loader, val_loader, test_loader, dataset


