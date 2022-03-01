import os
# import sys
# sys.path.append(os.path.join(os.getcwd(), '../..'))

from Code.network.resnet_backbone import LResNet, IR_Block
from Code.network.misc import NestedTensor, nested_tensor_from_tensor_list
from Code.network.position_encoding import build_position_encoding
from Code.dataset_helpers import get_data_loader, get_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision import transforms

from typing import List, Dict


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer3': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) # get intermediate values on specified layers
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, layers: list = [3, 4, 14, 3], filter_list: list = [64, 64, 128, 256, 512]):
        if name == "resnet50_ir":
            backbone = LResNet(IR_Block, layers = layers, filter_list = filter_list)
        else:
            pass
        
        num_channels = 256 # change to 512 if use last layer
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(layers: list = [3, 4, 14, 3], filter_list: list = [64, 64, 128, 256, 512], hidden_dim = 256, position_embedding = 'sine', train_backbone = False):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    backbone = Backbone("resnet50_ir", train_backbone, return_interm_layers = False, layers = layers, filter_list = filter_list)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model


# if __name__ == "__main__":
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
#     backbone = build_backbone()
#     backbone.to('cuda')
#     for data, label in casia_loader:
#         data, label = data.to('cuda'), label.to('cuda')
#         if isinstance(data, (list, torch.Tensor)):
#             data = nested_tensor_from_tensor_list(data)
#         out, pos = backbone(data)
#         print(out[-1].tensors.shape, pos[-1].shape)
    