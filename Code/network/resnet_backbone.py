import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary


class IR_Block(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, stride:int, match_dim:bool):
        """
        IR block implementation. One IR block consists of:
        """
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)        
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.prelu1 = nn.PReLU(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        if match_dim:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        """
        Forward propagation
        """
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class LResNet(nn.Module):
    """
    LResNet50-IR implementation.
    """

    def __init__(self, block, layers:list = [3, 4, 14, 3], filter_list:list = [64, 64, 128, 256, 512], is_gray:bool = False):
        """
        Initialization
        """
        super().__init__()

        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])

        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[-1] * 14 * 14),
            nn.Dropout(p = 0.4),
            nn.Linear(filter_list[-1] * 14 * 14, 512),
            nn.BatchNorm1d(512)
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel:int, out_channel:int, num_layers:int, stride:int):
        """
        """
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))

        for _ in range(1, num_layers):
            layers.append(block(out_channel, out_channel, stride=1, match_dim=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(batch_size, -1)
        x = self.fc(x)

        feature = {'features': x}
        additional_loss = {}
        additional_output = {}

        return feature, additional_loss, additional_output


class LResNet_Attention(LResNet):
    """
    """

    def __init__(self, block,
                 num_clusters:int, 
                 feature_dict_file:str, 
                 layers: list = [3, 4, 14, 3], filter_list: list = [64, 64, 128, 256, 512],
                 recurrent_step:int = 1,
                 normalize_feature_map:bool = False, use_threshold:bool = True, normalize_before_attention:bool = True,
                 is_gray: bool = False):
        """
        Initialization
        """
        super().__init__(block, layers = layers, filter_list = filter_list, is_gray = is_gray)

        self.num_clusters = num_clusters
        
        self.feature_dict_file = feature_dict_file

        self.recurrent_step = recurrent_step

        self.normalize_feature_map = normalize_feature_map
        self.use_threshold = use_threshold
        self.normalize_before_attention = normalize_before_attention

        # feature dictionary
        self.feature_dict = self._load_feature_dict()
        self.feature_dict.requires_grad = True

    def _load_feature_dict(self):
        """
        Feature dictionary pickle loader.

        :returns: Feature dictionary weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        print("\nLoad feature dictionary from:", self.feature_dict_file)
        with open(self.feature_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        assert self.num_clusters % feature_dict.shape[0] == 0, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_clusters} clusters!"

        if feature_dict.shape[0] != self.num_clusters:
            num_repeat = self.num_clusters // feature_dict.shape[0]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=0)

        if feature_dict.shape[1] != 256:
            num_repeat = 256 // feature_dict.shape[1]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=1)

        feature_dict = np.expand_dims(feature_dict, axis = [-1, -2]) # no need to transpose!
        print("Feature dictionary shape:", feature_dict.shape)
        
        return nn.Parameter(torch.tensor(feature_dict, dtype=torch.float32))
    
    def _clean_feature_dict(self):
        noise = [17, 27, 54, 57, 62]
        self.feature_dict.data[noise] = torch.zeros(*self.feature_dict.data[noise].shape)

    def load_feature_dict(self, feat_dict_file):
        """
        Manually load feature dictionary.

        :returns: Feature dictionary weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        print("\nLoad feature dictionary from:", feat_dict_file)
        with open(feat_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        assert self.num_clusters % feature_dict.shape[0] == 0, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_clusters} clusters!"

        if feature_dict.shape[0] != self.num_clusters:
            num_repeat = self.num_clusters // feature_dict.shape[0]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=0)

        if feature_dict.shape[1] != 256:
            num_repeat = 256 // feature_dict.shape[1]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=1)

        feature_dict = np.expand_dims(feature_dict, axis = [-1, -2]) # no need to transpose!
        print("Feature dictionary shape:", feature_dict.shape)
        
        self.feature_dict = nn.Parameter(torch.tensor(feature_dict, dtype=torch.float32))
        self.feature_dict.requires_grad = True

    def forward(self, x):
        """
        Forward propagation of LResNet50IR + Attention map. This consists of:
           1. Feed forward until stage-3 (second last layer)
           2. Normalize feature dictionary and feature map stage-3
           3. Compute similarity of feature map and feature dictionary using 1*1 convolution
           4. Compute & update the total of feature dictionary loss (vc_loss)
           5. Compute attention map
           6. Apply the attention map back to the pool1 layer
           7. Repeat 1-6 until specific number of recurrence (recurrence_step)
           8. Feedforward to stage-3 again, then compute the similarity, attention map, and vc_loss (similar to step 3-5)
           9. Feedforward to layer 4 & get the final attention map
           10.Feedforward into fully connected layer to get final 512 features
        """
        batch_size = x.shape[0]
        num_clusters = self.feature_dict.shape[0]
        num_channel = self.feature_dict.shape[1]
        attn_size = x.shape[-1] // 8

        vc_loss = 0
        attn_maps = []
        sim_clusters = []

        # forward pass until layer 3
        init_stage_result = self.conv1(x)
        init_stage_result = self.bn1(init_stage_result)
        init_stage_result = self.prelu1(init_stage_result)

        layer1_result = self.layer1(init_stage_result)
        layer2_result = self.layer2(layer1_result)
        layer3_result = self.layer3(layer2_result)

        # recurrence step
        for i in range(self.recurrent_step):
            norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
            layer3_norm = F.normalize(layer3_result, p=2, dim=1)

            # calculate similarity
            if self.normalize_feature_map:
                similarity = F.conv2d(layer3_norm, norm_feature_dict)
                similarity_norm = similarity
            else:
                similarity = F.conv2d(layer3_result, norm_feature_dict)
                similarity_norm = F.conv2d(layer3_norm, norm_feature_dict)

            cluster_assign = torch.max(similarity_norm, dim = 1).indices
            sim_clusters.append(cluster_assign.detach().clone())

            rshape_feat_dict = norm_feature_dict.reshape(num_clusters, num_channel)
            D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)

            # compute feature dictionary loss
            vc_loss += torch.mean(0.5 * (D - layer3_norm)**2)

            # get maximum similarity among all clusters -> max similarity == min distance
            attn = torch.max(similarity, dim=1).values # attention should be (28, 28)

            if self.use_threshold:
                attn_flat = attn.reshape(-1, attn_size * attn_size)
                percentage_l = 0.8
                percentage_u = 0.2
                percentage_ln = int(attn_size * attn_size * percentage_l)
                percentage_un = int(attn_size * attn_size * percentage_u)

                threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
                threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
                threshold_l = threshold_l.reshape(-1, 1, 1)
                threshold_u = threshold_u.reshape(-1, 1, 1)
                threshold_u = threshold_u.repeat(1, attn_size, attn_size)

                attn = F.relu(attn - threshold_l) + threshold_l
                attn = torch.min(torch.max(attn, torch.zeros_like(attn)), threshold_u)
                attn = attn / threshold_u
            else:
                attn = attn / torch.max(attn, dim=[2, 3], keepdim = True).values
            
            # store the attention map for visualization
            attn_maps.append(attn.detach().clone()) # 28 * 28

            # (N, C, 112, 112) -> (N, C, 14, 8, 14, 8)
            init_stage_result = init_stage_result.reshape(-1, 64, attn_size, 8, attn_size, 8)
            attn = attn.reshape(-1, 1, attn_size, 1, attn_size, 1)

            # apply attn on pool1
            new_init_stage = init_stage_result * attn
            new_init_stage = new_init_stage.reshape(-1, 64, 224, 224)

            # feed forward again
            layer1_result = self.layer1(new_init_stage)
            layer2_result = self.layer2(layer1_result)
            layer3_result = self.layer3(layer2_result)

        new_layer3 = layer3_result

        norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
        layer3_norm = F.normalize(new_layer3, p=2, dim=1)

        # calculate similarity
        if self.normalize_feature_map:
            similarity = F.conv2d(layer3_norm, norm_feature_dict)
            similarity_norm = similarity
        else:
            similarity = F.conv2d(new_layer3, norm_feature_dict)
            similarity_norm = F.conv2d(layer3_norm, norm_feature_dict)

        # feature dictionary loss
        cluster_assign = torch.max(similarity_norm, dim = 1).indices
        sim_clusters.append(cluster_assign.detach().clone())
        rshape_feat_dict = norm_feature_dict.reshape(num_clusters, num_channel)
        D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)
        vc_loss += torch.mean(0.5 * (D - layer3_norm)**2)

        # attention map
        attn_new = torch.max(similarity, dim=1, keepdim=True).values # attention should be (28, 28)
        if self.use_threshold:
            attn_flat = attn_new.reshape(-1, attn_size * attn_size)
            percentage_l = 0.8
            percentage_u = 0.2
            percentage_ln = int(attn_size * attn_size * percentage_l)
            percentage_un = int(attn_size * attn_size * percentage_u)

            threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
            threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
            threshold_l = threshold_l.reshape(-1, 1, 1, 1)
            threshold_u = threshold_u.reshape(-1, 1, 1, 1)
            threshold_u = threshold_u.repeat(1, 1, attn_size, attn_size)

            attn_new = F.relu(attn_new - threshold_l) + threshold_l
            attn_new = torch.min(torch.max(attn_new, torch.zeros_like(attn_new)), threshold_u)
            attn_new = attn_new / threshold_u
        else:
            attn_new = attn_new / torch.max(attn_new, dim=[2, 3], keepdim = True).values

        # store attention map for visualization
        attn_maps.append(attn_new.detach().clone())

        # forward propagation to layer 4
        layer4_result = self.layer4(layer3_result)
        attention = F.avg_pool2d(attn_new, (2,2), stride=2) # average pooling
        
        if self.normalize_before_attention:
            layer4_result = F.normalize(layer4_result, p=2, dim=[1,2,3]) # normalize C, H, W

        # apply filter and feed into fc layer
        features = layer4_result * attention
        features = features.reshape(batch_size, -1)
        features = self.fc(features)

        feature = {'features': features}
        additional_loss = {'vc_loss': vc_loss}
        additional_output = {
            'attn_map': attn_maps,
            'similar_cluster': sim_clusters,
        }

        return feature, additional_loss, additional_output


def initialize_LResNet50_IR(is_gray:bool = False):
    """
    Function to get original LResNet50-IR model. Some hyperparameters are choosen in advance.

    :param is_gray: Whether images is in grayscale format

    :returns: LResNet50-IR model
    :rtype:   LResNet
    """
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    
    return LResNet(IR_Block, layers, filter_list, is_gray)


def initialize_LResNet50_attn(num_clusters:int, feature_dict_file:str, recurrent_step:int = 1):
    """
    Function to get original LResNet50-IR + Attention model. Some hyperparameters are choosen in advance.

    :param num_clusters:      Number of clusters in feature dictionary
    :param feature_dict_file: Path to the initial feature dictionary weight (vMF clusters)
    :param recurrent_step:    Number of reccurence to filter out the occluded features

    :returns: LResNet50-IR + Attention model
    :rtype:   LResNet_Attention
    """
    return LResNet_Attention(
        IR_Block,
        num_clusters,
        feature_dict_file,
        recurrent_step = recurrent_step 
    )


if __name__ == "__main__":
    feature_dict_file = "feature_dictionary/CASIA/100000/dictionary_second_64.pickle"
    model = initialize_LResNet50_attn(64, feature_dict_file)

    summary(model, (3, 224, 224), device = "cpu")