import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), '../..'))
# from Code.dataset_helpers import get_dataset, get_data_loader
import logging
logger = logging.getLogger(__name__)
print = logger.info

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import transforms

from Code.network.detr_backbone import build_backbone
from Code.network.misc import nested_tensor_from_tensor_list
from Code.network.transformer import build_transformer


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_queries, out_features, queries_weight = ""):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.queries_weight = queries_weight
        self.hidden_dim = transformer.d_model
        
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        if os.path.exists(self.queries_weight):
            self.query_embed.weight = self._load_queries_weight()
            self.query_embed.weight.requires_grad = True
        else:
            print("Queries weight not found!")

        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size = 1)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_queries * self.hidden_dim),
            nn.Dropout(p = 0.2),
            nn.Linear(num_queries * self.hidden_dim, out_features),
            nn.BatchNorm1d(out_features)
        )

    def _load_queries_weight(self):
        """
        Feature dictionary pickle loader.

        :returns: Feature dictionary weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        print(f"Load feature dictionary from: {self.queries_weight}")
        with open(self.queries_weight, 'rb') as file:
            feature_dict = pickle.load(file)

        assert self.num_queries % feature_dict.shape[0] == 0, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_queries} queries!"

        # Match dimension with query embedding weight
        if feature_dict.shape[0] != self.num_queries:
            num_repeat = self.num_queries // feature_dict.shape[0]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=0)

        if feature_dict.shape[1] != self.hidden_dim:
            num_repeat = self.hidden_dim // feature_dict.shape[1]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=1)

        print(f"Feature dictionary shape: {feature_dict.shape}")
        
        return nn.Parameter(torch.tensor(feature_dict, dtype = torch.float32))

    def forward(self, x):
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)
        
        feature_map, pos = self.backbone(x)
        src, mask = feature_map[-1].decompose()
        assert mask is not None

        out = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        output = self.fc(out[-1].flatten(1))

        features = {'features': output}
        additional_loss = {}
        additional_output = {"attn_map": [None]}
        
        return features, additional_loss, additional_output


def build_model(out_features, layers: list = [3, 4, 14, 3], filter_list: list = [64, 64, 128, 256, 512], backbone_weight = "",
                hidden_dim = 256, position_embedding = 'sine', train_backbone = False, 
                num_encoder_layers = 6, num_decoder_layers = 6, return_interm_result = True, 
                num_queries = 64, queries_weight = ""):

    backbone = build_backbone(
        layers= layers,
        filter_list=filter_list,
        backbone_weight = backbone_weight,
        hidden_dim = hidden_dim,
        position_embedding = position_embedding,
        train_backbone = train_backbone
    )

    transformer = build_transformer(
        hidden_dim = hidden_dim,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        return_iterm = return_interm_result
    )

    model = DETR(backbone, transformer, num_queries, out_features, queries_weight)

    return model


# if __name__ == '__main__':
#     root = "../../dataset"
#     annot = "CASIA_aligned_list.txt"
#     data = "CASIA_aligned"

#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
#     ])

#     dataset, _ = get_dataset(
#         root,
#         data,
#         annot,
#         transform,
#         mode = 'annotation'
#     )

#     casia_loader = get_data_loader(dataset, batch_size = 8)
#     model = build_model(512)
#     model.to('cuda')

#     for data, label in casia_loader:
#         data, label = data.to('cuda'), label.to('cuda')
#         features, additinal_loss, additional_output = model(data)
#         print(features['features'].shape)
#         model.zero_grad()
        