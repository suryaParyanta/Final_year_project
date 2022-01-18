import os

import pickle
import logging
logger = logging.getLogger(__name__)
print = logger.info

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary


class Conv_Block(nn.Module):
    """
    Convolution Block implementation. One block consists of:
        Conv layer -> ReLU
    The learnable weights are initialized using Kaiming (He) normal initialization.
    """

    def __init__(self, in_channel:int, out_channel:int, stride:int = 1):
        """
        Convolution block initialization.

        :param in_channel:  Input channel of convolution layer
        :param out_channel: Output channel of convolution layer
        :param stride:      Stride of convolution
        """
        super(Conv_Block, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward propagation of convolution block. It consists of:
            1. Convolution Layer
            2. Activation function (ReLU)
        
        :param x: Tensor of size (batch_size, in_channel, H, W)
        :type x:  torch.FloatTensor
        :returns: A tensor with size (batch_size, out_channel, HH, WW)
        """
        x = self.conv(x)
        x = self.relu(x)

        return x


class View(nn.Module):
    """
    View layer implementation. This layer will reshape the flattened feature map 
    into 2D with specified size.
    """

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        """
        View layer forward propagation. Reshape the input tensor into specified shape.

        :param x: Input tensor of shape (batch_size, C, H * W)
        :type x:  torch.FloatTensor

        :returns: A reshaped tensor with shape (batch_size, C, H, W)
        :rtype:   torch.FloatTensor
        """
        bs, c_in = x.shape[0:2]
        target_hw = int(round(x.shape[-1] ** 0.5, 0))

        return x.reshape(bs, c_in, target_hw, target_hw)


class Identity(nn.Module):
    """
    Identity layer implementation. This layer will only pass the input tensor into the next layer.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x    

        
def init_weights(m):
    """
    Kaiming weight initialization function for Linear Layer. Apply this function into pytorch nn.Sequential.
    """
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')


class VGG_16(nn.Module):
    """
    Implementation of VGG16 model. This model consists of five stages (layers):
       - layer1: ( Conv - ReLU ) × 2 - MaxPool,   Input: (batch_size, C, H, W)          Output: (batch_size, 64, H/2, W/2)    
       - layer2: ( Conv - ReLU ) × 2 - MaxPool,   Input: (batch_size, 64, H/2, W/2)     Output: (batch_size, 128, H/4, W/4)
       - layer3: ( Conv - ReLU ) × 3 - MaxPool,   Input: (batch_size, 128, H/4, W/4)    Output: (batch_size, 256, H/8, W/8)
       - layer4: ( Conv - ReLU ) × 3 - MaxPool,   Input: (batch_size, 256, H/8, W/8)    Output: (batch_size, 512, H/16, W/16)
       - layer5: ( Conv - ReLU ) × 3 - MaxPool,   Input: (batch_size, 512, H/16, W/16)  Output: (batch_size, 512, H/32, W/32)
    """

    def __init__(self, block, layers:list = [2,2,3,3,3], filter_list:list = [64,128,256,512,512], num_classes:int = 10, is_gray:bool = False):
        """
        VGG16 Initialization.

        :param block:       Basic building block for VGG16
        :param layers:      Number of neural network layers for each stage in VGG16
        :param filter_list: Number of Convolution filters at each stage
        :param num_classes: Number of classes in the dataset
        :param is_gray:     Whether the dataset images is in grayscale format
        """
        super().__init__()

        if is_gray:
          self.layer1 = self._vgg_block(block, 1, filter_list[0], layers[0], stride=1)
        else:
          self.layer1 = self._vgg_block(block, 3, filter_list[0], layers[0], stride=1)

        self.layer2 = self._vgg_block(block, filter_list[0], filter_list[1], layers[1], stride=1)
        self.layer3 = self._vgg_block(block, filter_list[1], filter_list[2], layers[2], stride=1)
        self.layer4 = self._vgg_block(block, filter_list[2], filter_list[3], layers[3], stride=1)
        self.layer5 = self._vgg_block(block, filter_list[3], filter_list[4], layers[4], stride=1)

        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=True)
        )
        self.fc.apply(init_weights)
    
    def _vgg_block(self, block, in_channel:int, out_channel:int, num_layers:int, stride:int = 1):
        """
        VGG16 one stage implementation.

        :param block:       Basic building block for VGG16
        :param in_channel:  Input channel that will be feed into this stage
        :param out_channel: Output channel of this stage
        :param num_layers:  Number of Convolution layers
        :param stride:      Stride for convolution layers
        
        :returns: One combined block that consists of multiple building blocks
        :rtype:   torch.nn.Sequential
        """
        layers = []
        layers.append(block(in_channel, out_channel, stride))

        for _ in range(1, num_layers):
            layers.append(block(out_channel, out_channel, stride))
        layers.append(nn.MaxPool2d(2,2))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation of VGG16 and then pass the feature map into fully connected layer.

        :param x: Image minibatch; tensor of shape (batch_size, C, H, W)
        
        :results:  Tuple of dictionaries that consists of:
                      1. Scores (logits)
                      2. Additional loss (for VGG with attention & prototype) 
                      3. Intermediate output (for visualization)
        :rtype:    tuple
        """
        bs = x.shape[0]

        # convolution layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # fc layer
        x = x.reshape(bs, -1)
        x = self.fc(x)

        # define output dictionaries
        score = {
            'score': x
        }
        additional_loss = {}
        additional_output = {}

        return score, additional_loss, additional_output


class VGG_Attention_Prototype(VGG_16):
    """
    Implementation of TDMPNet/TDAPNet model (https://arxiv.org/abs/1909.03879). The backbone of this model is VGG16, however there is an additional learnable parameter 
    after pool4 of VGG16, which is called feature dictionary. This parameter will be initalized using vMF clustering that can be trained 
    using the following github repo (https://github.com/AdamKortylewski/CompositionalNets). To handle partial image occlusion, this parameter
    will help the feature map to filter out the occluded features.

    Another additional layer called prototype is created to replace the fully-connected layer in original VGG16. This prototype layer will also help the model
    to be robust against occlusion by capturing large changes in spatial patterns (ex: different viewpoints). The prototype weights are initialized using K-means
    clustering on the dataset.
    """
    
    def __init__(self, block, 
                 num_classes:int, num_prototype:int, num_clusters:int,
                 prototype_file:str, feature_dict_file:str,
                 layers:list = [2,2,3,3,3], filter_list:list = [64,128,256,512,512], 
                 prototype_method:str = "kmeans", recurrent_step:int = 1,
                 percent_u:float = 0.2, percent_l:float = 0.8,
                 normalize_feature_map:bool = False, use_threshold:bool = True, normalize_before_attention:bool = True, 
                 distance:str = 'euclidean',
                 is_gray:bool = False):
        """
        Initialization of TDMPNet.

        :param block:                      Basic building block of the model
        :param num_classes:                Number of classes in the dataset
        :param num_prototype:              Number of prototype for each classes
        :param num_clusters:               Number of clusters in feature dictionary
        :param prototype_file:             Path to the initial prototype weight (K-means clusters)
        :param feature_dict_file:          Path to the initial feature dictionary weight (vMF clusters)
        :param layers:                     Number of neural network layers for each stage
        :param filter_list:                Number of convolution filters at each stage
        :param recurrent_step:             Number of reccurence to filter out the occluded features 
        :param normalize_feature_map:      Whether to normalize feature map
        :param use_threshold:              Whether to do clipping on attention map
        :param normalize_before_attention: Whether to normalize prototype weight and pool5 feature map
        :param distance:                   Metrics to compute distance
        :param is_gray:                    Whether the dataset images is in grayscale format
        """
        super().__init__(block, layers = layers, filter_list = filter_list, num_classes = num_classes, is_gray = is_gray)

        self.prototype_file = prototype_file
        self.feature_dict_file = feature_dict_file

        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.num_clusters = num_clusters

        self.prototype_method = prototype_method
        self.recurrent_step = recurrent_step

        self.percent_u = percent_u
        self.percent_l = percent_l

        self.normalize_feature_map = normalize_feature_map
        self.use_threshold = use_threshold
        self.normalize_before_attention = normalize_before_attention

        self.distance = distance
        
        # set own learnable parameter
        self.feature_dict = self._load_feature_dict()
        self.feature_dict.requires_grad = True

        self.gamma = nn.Parameter(torch.tensor([20.]))
        self.gamma.requires_grad = True

        self.fc = Identity() # remove fully connected layer
        self.prototype_weight = self._load_prototype_weight()
        self.prototype_weight.requires_grad = True
    
    def _load_feature_dict(self):
        """
        Feature dictionary pickle loader.

        :returns: Feature dictionary weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        print(f"Load feature dictionary from: {self.feature_dict_file}")
        with open(self.feature_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        assert feature_dict.shape[0] == self.num_clusters, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_clusters} clusters!"

        feature_dict = np.expand_dims(feature_dict, axis = [-1, -2]) # no need to transpose!
        print(f"Feature dictionary shape: {feature_dict.shape}")

        return nn.Parameter(torch.tensor(feature_dict, dtype=torch.float32))

    def _load_prototype_weight(self):
        """
        Prototype weight pickle loader.

        :returns: Prototype weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        if not os.path.exists(self.prototype_file):
            assert self.prototype_method in ["kmeans", "random"], "Unsupported prototype methods"

            if self.prototype_method == "random":
                print(f"Prototype weight not found! Use random initialization instead...")
                proto_weight = nn.Parameter(torch.zeros(self.num_classes * self.num_prototype, 512, 7, 7, dtype = torch.float32))
                init.xavier_uniform_(proto_weight)

                return proto_weight
            
            print("")
            print(f"Training prototype with kmeans...")
            proto_weight = self._initialize_prototype()

        elif os.path.exists(self.prototype_file):
            print(f"Load Prototype weight from: {self.prototype_file}")

            with open(self.prototype_file, 'rb') as file:
                proto_weight = pickle.load(file)
                
        *_, C, H, W = proto_weight.shape
        proto_weight = proto_weight.reshape(-1, C, H, W)
        assert proto_weight.shape[0] == self.num_classes * self.num_prototype, f"Prototype weight dim 0 does not equal to {self.num_classes * self.num_prototype}!"
        
        print(f"Prototype weight shape: {proto_weight.shape}")

        return nn.Parameter(torch.tensor(proto_weight, dtype=torch.float32))
    
    def _initialize_prototype(self):
        """
        Generate initial prototype weight. The process involves:
        1. Forward propagation to obtain feature map
        2. For each class, perform K-means clustering based on the feature map from step-1

        :returns: Prototype weight stored on the pickle file
        """
        import sys
        import gc
        sys.path.append(os.getcwd())
        from sklearn.cluster import KMeans
        from Code.dataset_helpers import get_dataset, get_data_loader, DatasetSampler

        batch_size = 128;
        proto_weight = np.ndarray((self.num_classes, self.num_prototype, 512, 7, 7)).astype(np.float32)

        # load pretrained MNIST VGG
        vgg_pretrain_path = "pretrained_weight/VGG_MNIST/best.pt"
        assert os.path.exists(vgg_pretrain_path), f"{vgg_pretrain_path} not exists"

        pretrain_vgg = initialize_vgg(self.num_classes)
        print(f"Load pretrain VGG from: {vgg_pretrain_path}")
        pretrain_vgg.load_state_dict(torch.load(vgg_pretrain_path))

        # freeze the layers and remove fully connected layer
        for p in pretrain_vgg.parameters():
            p.requires_grad = False
        pretrain_vgg.fc = Identity()

        # load dataset
        train_root = "dataset/MNIST_224X224_3"
        train_annot = "pairs_train.txt"
        train_data = "train"
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset, _ = get_dataset(train_root, train_data, train_annot, transform)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrain_vgg.to(device)
        pretrain_vgg.eval()

        for i in range(self.num_classes):
            mask = [1 if label == i else 0 for _, label in train_dataset]
            num_images = sum(mask)
            pool5_features = np.ndarray((num_images, 512*7*7)).astype(np.float32)

            # prepare the DataLoader
            mask = torch.tensor(mask)   
            sampler = DatasetSampler(mask, train_dataset)
            train_loader = get_data_loader(train_dataset, batch_size = batch_size, shuffle = False, sampler = sampler)

            for batch_idx, (data, label) in enumerate(train_loader):
                idx_start = batch_idx * batch_size
                idx_end = min([(batch_idx+1) * batch_size, num_images])
                
                # forward propagation
                with torch.no_grad():
                    data = data.to(device)
                    feature_map, *_ = pretrain_vgg(data)
                    feature_map = feature_map["score"].to('cpu').detach().numpy()
                
                pool5_features[idx_start:idx_end, :] = feature_map

            kmeans = KMeans(n_clusters = self.num_prototype)
            kmeans.fit(pool5_features)
            centers = kmeans.cluster_centers_

            centers = np.reshape(centers, (self.num_prototype, 512, 7, 7))
            proto_weight[i, :, :, :, :] = centers

        save_path = os.path.join("prototype_weight", f"prototype_{self.num_classes}_{self.num_prototype}_MNIST.pkl")
        if os.path.exists(save_path):
            idx = 2
            while os.path.exists(save_path.split(".")[0] + f"_{idx}" + ".pkl"):
                idx += 1
            save_path = save_path.split(".")[0] + f"_{idx}" + ".pkl"

        print(f"Saving prototype weight into: {save_path}")
        with open(save_path, 'wb') as prototype_file:
            pickle.dump(proto_weight, prototype_file)

        # delete unnecessary variable
        del train_loader, train_dataset
        gc.collect();

        return proto_weight

    def forward(self, x):
        """
        Forward propagation of TDMPNet. This consists of:
           1. Feed forward until stage-4 (after pool4 layer) of backbone VGG16
           2. Normalize feature dictionary and feature map stage-4
           3. Compute similarity of feature map and feature dictionary using 1*1 convolution
           4. Compute & update the total of feature dictionary loss (vc_loss)
           5. Compute attention map
           6. Apply the attention map back to the pool1 layer
           7. Repeat 1-6 until specific number of recurrence (recurrence_step)
           8. Feedforward to stage-4 again, then compute the similarity, attention map, and vc_loss (similar to step 3-5)
           9. Feedforward to layer 5 & get the attention map by avgpool
           10.Feedforward to prototype layer, then compute logits and prototype loss
        
        :param x: Image minibatch; tensor of shape (batch_size, C, H, W)
        
        :results:  Tuple of dictionaries that consists of:
                      1. Scores (logits)
                      2. Additional loss (vc_loss & prototype_loss) 
                      3. Intermediate output (attention map after pool4 layer)
        :rtype:    tuple
        """
        batch_size = x.shape[0]
        num_clusters = self.feature_dict.shape[0]
        vc_loss = 0

        # forward pass until layer 4
        pool1_result = self.layer1(x)
        pool2_result = self.layer2(pool1_result)
        pool3_result = self.layer3(pool2_result)
        pool4_result = self.layer4(pool3_result)

        # recurrence step
        for i in range(self.recurrent_step):
            norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
            pool4_norm = F.normalize(pool4_result, p=2, dim=1)

            # calculate similarity
            if self.normalize_feature_map:
                similarity = F.conv2d(pool4_norm, norm_feature_dict)
                similarity_norm = similarity
            else:
                similarity = F.conv2d(pool4_result, norm_feature_dict)
                similarity_norm = F.conv2d(pool4_norm, norm_feature_dict)

            # get the most similar cluster 
            cluster_assign = torch.max(similarity_norm, dim=1).indices

            # reshape (Cout, Cin, 1, 1) to (Cout, Cin)
            # Cout: num of cluster cluster, Cin: depth dimension
            rshape_feat_dict = norm_feature_dict.reshape(num_clusters,512)
            D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)

            # compute feature dictionary loss
            vc_loss += torch.mean(0.5 * (D - pool4_norm)**2)

            # get maximum similarity among all clusters -> max similarity == min distance
            attn = torch.max(similarity, dim=1).values

            # clipping the attention map
            if self.use_threshold:
                attn_flat = attn.reshape(-1, 14 * 14)
                percentage_ln = int(14 * 14 * self.percent_l)
                percentage_un = int(14 * 14 * self.percent_u)

                threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
                threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
                threshold_l = threshold_l.reshape(-1,1,1)
                threshold_u = threshold_u.reshape(-1,1,1)
                threshold_u = threshold_u.repeat(1, 14, 14)

                attn = F.relu(attn - threshold_l) + threshold_l
                attn = torch.min(torch.max(attn, torch.zeros_like(attn)), threshold_u)
                attn = attn / threshold_u
            else:
                attn = attn / torch.max(attn, dim=[2, 3], keepdim=True).values
            
            # (N, C, 112, 112) -> (N, C, 14, 8, 14, 8)
            pool1_result = pool1_result.reshape(-1, 64, 14, 8, 14, 8)
            attn = attn.reshape(-1, 1, 14, 1, 14, 1)

            # apply attn on pool1
            new_pool1 = pool1_result * attn
            new_pool1 = new_pool1.reshape(-1, 64, 112, 112)

            pool2_result = self.layer2(new_pool1)
            pool3_result = self.layer3(pool2_result)
            pool4_result = self.layer4(pool3_result)

        new_pool4 = pool4_result

        norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
        pool4_norm = F.normalize(new_pool4, p=2, dim=1)

        # calculate similarity
        if self.normalize_feature_map:
            similarity = F.conv2d(pool4_norm, norm_feature_dict)
            similarity_norm = similarity
        else:
            similarity = F.conv2d(new_pool4, norm_feature_dict)
            similarity_norm = F.conv2d(pool4_norm, norm_feature_dict)
        
        # feature dictionary loss
        cluster_assign = torch.max(similarity_norm, dim=1).indices
        rshape_feat_dict = norm_feature_dict.reshape(num_clusters,512)
        D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)
        vc_loss += torch.mean(0.5 * (D - pool4_norm)**2)

        attn_new = torch.max(similarity, dim=1, keepdim=True).values
        if self.use_threshold:
            attn_flat = attn_new.reshape(-1, 14 * 14)
            percentage_ln = int(14 * 14 * self.percent_l)
            percentage_un = int(14 * 14 * self.percent_u)

            threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
            threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
            threshold_l = threshold_l.reshape(-1,1,1,1)
            threshold_u = threshold_u.reshape(-1,1,1,1)
            threshold_u = threshold_u.repeat(1, 1, 14, 14)

            # get attention map
            attn_new = F.relu(attn_new - threshold_l) + threshold_l
            attn_new = torch.min(torch.max(attn_new, torch.zeros_like(attn_new)), threshold_u)
            attn_new = attn_new / threshold_u
        else:
            attn_new = attn_new / torch.max(attn_new, dim=[2, 3], keepdim=True).values
        
        # for visualization
        attn_pool4 = attn_new.detach().clone()

        # forward propagation to layer 5
        pool5_result = self.layer5(pool4_result)
        attention = F.avg_pool2d(attn_new, (2,2), stride=2) # average pooling
        prototype = self.prototype_weight
        
        if self.normalize_before_attention:
            pool5_result = F.normalize(pool5_result, p=2, dim=[1,2,3]) # normalize C, H, W
            prototype = F.normalize(self.prototype_weight, p=2, dim=[1,2,3]) # normalize C, H, W
        
        features = pool5_result * attention
        prototype = prototype.reshape(1, *prototype.shape)
        prototype = prototype.repeat(batch_size, 1, 1, 1, 1)
        attention = attention.reshape(-1, 1, 1, 7, 7)
        prototype = attention * prototype

        # calculate distance
        if self.distance == 'euclidean':
            features = features.reshape(-1, 512 * 7 * 7, 1)
            prototype = prototype.reshape(-1,self.num_classes * self.num_prototype, 512 * 7 * 7)
            sq_l2_features = torch.sum(features**2, dim=[1,2], keepdim=True)
            sq_l2_prototype = torch.sum(prototype**2, dim=2, keepdim=True)

            distances = -1 * (sq_l2_features + sq_l2_prototype - 2 * torch.matmul(prototype, features))
            distances = distances.reshape(-1, self.num_classes * self.num_prototype)
            distance_class = torch.max(distances.reshape(-1, self.num_classes, self.num_prototype), dim=2).values
            prediction_idx = torch.max(distances, dim=1).indices
        
        # calculate prototype loss
        pred_idx = prediction_idx
        batch_idx = torch.arange(0, batch_size).to(pred_idx.device)
        assigned_prototype = prototype[batch_idx, pred_idx]
        features = features.reshape(-1, 7 * 7 * 512)
        prototype_loss = torch.mean(torch.sum((features - assigned_prototype)**2, dim=1) / 2)

        logits = self.gamma * distance_class
        score = {'score': logits}

        additional_loss = {
            'prototype_loss': prototype_loss,
            'vc_loss': vc_loss
        }

        additional_output = {
            'attn_pool4': attn_pool4,
            'distance_class': distance_class.detach().clone(),
            'assigned_prototype': assigned_prototype.detach().clone()
        }
        return score, additional_loss, additional_output


def xavier_uniform_init(m):
    """
    Xavier uniform random initialization. Apply this function into nn.Sequential class.
    """
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)


class VGG_Attention(VGG_16):
    """
    Modification of TDMPNet model. This model do not have prototype weight as learnable parameter.
    Instead, this model will output 512 features that will be used for face recognition task. 
    """

    def __init__(self, block,
                 num_clusters:int,
                 feature_dict_file:str,
                 layers:list = [2,2,3,3,3], filter_list:list = [64,128,256,512,512], 
                 recurrent_step:int = 1,
                 percent_u:float = 0.2, percent_l:float = 0.8,
                 normalize_feature_map:bool = False, use_threshold:bool = True, normalize_before_attention:bool = True,
                 is_gray:bool = False):
        """
        Initialization of VGG16 + Attention map.

        :param block:                      Basic building block of the model
        :param num_clusters:               Number of clusters in feature dictionary
        :param feature_dict_file:          Path to the initial feature dictionary weight (vMF clusters)
        :param layers:                     Number of neural network layers for each stage
        :param filter_list:                Number of convolution filters at each stage
        :param recurrent_step:             Number of reccurence to filter out the occluded features 
        :param normalize_feature_map:      Whether to normalize feature map
        :param use_threshold:              Whether to do clipping on attention map
        :param normalize_before_attention: Whether to normalize prototype weight and pool5 feature map
        :param is_gray:                    Whether the dataset images is in grayscale format
        """
        super().__init__(block, layers = layers, filter_list = filter_list, is_gray = is_gray)

        self.num_clusters = num_clusters

        self.feature_dict_file = feature_dict_file
        
        self.recurrent_step = recurrent_step

        self.percent_u = percent_u
        self.percent_l = percent_l

        self.normalize_feature_map = normalize_feature_map
        self.use_threshold = use_threshold
        self.normalize_before_attention = normalize_before_attention
        
        # feature dictionary
        self.feature_dict = self._load_feature_dict()
        self.feature_dict.requires_grad = True

        # fully connected layer
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[-1] * 7 * 7),
            nn.Dropout(),
            nn.Linear(filter_list[-1] * 7 * 7, 512),
            nn.BatchNorm1d(512)
        )
        self.fc.apply(xavier_uniform_init)
    
    def _load_feature_dict(self):
        """
        Feature dictionary pickle loader.

        :returns: Feature dictionary weight as learnable parameter
        :rtype:   torch.nn.Parameter
        """
        print(f"Load feature dictionary from: {self.feature_dict_file}")
        with open(self.feature_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        assert self.num_clusters % feature_dict.shape[0] == 0, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_clusters} clusters!"

        if feature_dict.shape[0] != self.num_clusters:
            num_repeat = self.num_clusters // feature_dict.shape[0]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=0)

        if feature_dict.shape[1] != 512:
            num_repeat = 512 // feature_dict.shape[1]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=1)

        feature_dict = np.expand_dims(feature_dict, axis = [-1, -2]) # no need to transpose!
        print(f"Feature dictionary shape: {feature_dict.shape}")
        
        return nn.Parameter(torch.tensor(feature_dict, dtype=torch.float32))

    def load_feature_dict(self, feat_dict_file):
        """
        Manually load pretrained feature dictionary (for evaluation).

        :param feature_dict_file: Path to the pre-trained feature dictionary. 
        """
        print(f"Load feature dictionary manually from: {feat_dict_file}")
        with open(feat_dict_file, 'rb') as file:
            feature_dict = pickle.load(file)

        assert self.num_clusters % feature_dict.shape[0] == 0, f"Cannot fit feature dictionary with {feature_dict.shape[0]} clusters into {self.num_clusters} clusters!"

        if feature_dict.shape[0] != self.num_clusters:
            num_repeat = self.num_clusters // feature_dict.shape[0]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=0)

        if feature_dict.shape[1] != 512:
            num_repeat = 512 // feature_dict.shape[1]
            feature_dict = np.repeat(feature_dict, num_repeat, axis=1)

        feature_dict = np.expand_dims(feature_dict, axis = [-1, -2]) # no need to transpose!
        print(f"Feature dictionary shape: {feature_dict.shape}")

        self.feature_dict = nn.Parameter(torch.tensor(feature_dict, dtype=torch.float32))
        self.feature_dict.requires_grad = True

    def forward(self, x):
        """
        Forward propagation of VGG16 + Attention map. This consists of:
           1. Feed forward until stage-4 (after pool4 layer) of backbone VGG16
           2. Normalize feature dictionary and feature map stage-4
           3. Compute similarity of feature map and feature dictionary using 1*1 convolution
           4. Compute & update the total of feature dictionary loss (vc_loss)
           5. Compute attention map
           6. Apply the attention map back to the pool1 layer
           7. Repeat 1-6 until specific number of recurrence (recurrence_step)
           8. Feedforward to stage-4 again, then compute the similarity, attention map, and vc_loss (similar to step 3-5)
           9. Feedforward to layer 5 & get the attention map by avgpool
           10.Feedforward into fully connected layer to get final 512 features
        
        :param x: Image minibatch; tensor of shape (batch_size, C, H, W)
        
        :results:  Tuple of dictionaries that consists of:
                      1. Features
                      2. Additional loss (vc_loss) 
                      3. Intermediate output (attention map after pool4 layer)
        :rtype:    tuple
        """
        batch_size = x.shape[0]
        num_clusters = self.feature_dict.shape[0]
        
        vc_loss = 0
        attn_maps = []
        sim_clusters = []

        # forward pass until layer 4
        pool1_result = self.layer1(x)
        pool2_result = self.layer2(pool1_result)
        pool3_result = self.layer3(pool2_result)
        pool4_result = self.layer4(pool3_result)

        # recurrence step
        for i in range(self.recurrent_step):
            norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
            pool4_norm = F.normalize(pool4_result, p=2, dim=1)

            # calculate similarity
            if self.normalize_feature_map:
                similarity = F.conv2d(pool4_norm, norm_feature_dict)
                similarity_norm = similarity
            else:
                similarity = F.conv2d(pool4_result, norm_feature_dict)
                similarity_norm = F.conv2d(pool4_norm, norm_feature_dict)

            cluster_assign = torch.max(similarity_norm, dim = 1).indices
            sim_clusters.append(cluster_assign.detach().to('cpu').clone())

            rshape_feat_dict = norm_feature_dict.reshape(num_clusters, 512)
            D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)

            # compute feature dictionary loss
            vc_loss += torch.mean(0.5 * (D - pool4_norm)**2)

            # get maximum similarity among all clusters -> max similarity == min distance
            attn = torch.max(similarity, dim=1).values

            if self.use_threshold:
                attn_flat = attn.reshape(-1, 14 * 14)
                percentage_ln = int(14 * 14 * self.percent_l)
                percentage_un = int(14 * 14 * self.percent_u)

                threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
                threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
                threshold_l = threshold_l.reshape(-1,1,1)
                threshold_u = threshold_u.reshape(-1,1,1)
                threshold_u = threshold_u.repeat(1, 14, 14)

                attn = F.relu(attn - threshold_l) + threshold_l
                attn = torch.min(torch.max(attn, torch.zeros_like(attn)), threshold_u)
                attn = attn / threshold_u
            else:
                attn = attn / torch.max(attn, dim=[2, 3], keepdim = True).values
            
            # store the attention map for visualization
            attn_maps.append(attn.detach().to('cpu').clone())

            # (N, C, 112, 112) -> (N, C, 14, 8, 14, 8)
            pool1_result = pool1_result.reshape(-1, 64, 14, 8, 14, 8)
            attn = attn.reshape(-1, 1, 14, 1, 14, 1)

            # apply attn on pool1
            new_pool1 = pool1_result * attn
            new_pool1 = new_pool1.reshape(-1, 64, 112, 112)

            # feed forward again
            pool2_result = self.layer2(new_pool1)
            pool3_result = self.layer3(pool2_result)
            pool4_result = self.layer4(pool3_result)

        new_pool4 = pool4_result

        norm_feature_dict = F.normalize(self.feature_dict, p=2, dim=1)
        pool4_norm = F.normalize(new_pool4, p=2, dim=1)

        # calculate similarity
        if self.normalize_feature_map:
            similarity = F.conv2d(pool4_norm, norm_feature_dict)
            similarity_norm = similarity
        else:
            similarity = F.conv2d(new_pool4, norm_feature_dict)
            similarity_norm = F.conv2d(pool4_norm, norm_feature_dict)
        
        # feature dictionary loss
        cluster_assign = torch.max(similarity_norm, dim=1).indices
        sim_clusters.append(cluster_assign.detach().to('cpu').clone()) # for visualization
        rshape_feat_dict = norm_feature_dict.reshape(num_clusters, 512)
        D = rshape_feat_dict[cluster_assign].permute(0,3,1,2)
        vc_loss += torch.mean(0.5 * (D - pool4_norm)**2)

        # attention map
        attn_new = torch.max(similarity, dim=1, keepdim=True).values
        if self.use_threshold:
            attn_flat = attn_new.reshape(-1, 14 * 14)
            percentage_ln = int(14 * 14 * self.percent_l)
            percentage_un = int(14 * 14 * self.percent_u)

            threshold_l = torch.topk(attn_flat, percentage_ln, dim=1).values[:, -1] #.values get the values (not index) -> size (N, percentage_l)
            threshold_u = torch.topk(attn_flat, percentage_un, dim=1).values[:, -1]
            threshold_l = threshold_l.reshape(-1,1,1,1)
            threshold_u = threshold_u.reshape(-1,1,1,1)
            threshold_u = threshold_u.repeat(1, 1, 14, 14)

            attn_new = F.relu(attn_new - threshold_l) + threshold_l
            attn_new = torch.min(torch.max(attn_new, torch.zeros_like(attn_new)), threshold_u)
            attn_new = attn_new / threshold_u
        else:
            attn_new = attn_new / torch.max(attn_new, dim=[2, 3], keepdim=True).values
        
        # store attention map for visualization
        attn_maps.append(attn_new.detach().to('cpu').clone())

        # forward propagation to layer 5
        pool5_result = self.layer5(pool4_result)
        attention = F.avg_pool2d(attn_new, (2,2), stride=2) # average pooling
        
        if self.normalize_before_attention:
            pool5_result = F.normalize(pool5_result, p=2, dim=[1,2,3]) # normalize C, H, W
        
        # apply filter and feed into fc layer
        features = pool5_result * attention
        features = features.reshape(batch_size, -1)
        features = self.fc(features)

        feature = {'features': features}
        additional_loss = {'vc_loss': vc_loss}
        additional_output = {
            'attn_map': attn_maps,
            'similar_cluster': sim_clusters,
        }

        return feature, additional_loss, additional_output


def initialize_vgg(num_classes:int = 10):
    """
    Function to get original VGG16 model. Some hyperparameters are choosen in advance.

    :param num_classes: Number of classes in the dataset

    :returns: VGG16 model
    :rtype:   VGG_16
    """
    layers = [2,2,3,3,3]
    filter_list = [64,128,256,512,512]

    return VGG_16(Conv_Block, layers, filter_list, num_classes)


def initialize_vgg_attn_prototype(
    num_classes:int, num_prototype:int, num_clusters:int, 
    prototype_file:str, feature_dict_file:str, 
    prototype_method:str = "kmeans", recurrent_step:int = 1, 
    percent_u:float = 0.2, percent_l:float = 0.8):
    """
    Function to get original TDMPNet/TDAPNet. Some hyperparameters are choosen in advance.

    :param num_classes:       Number of classes in the dataset
    :param num_prototype:     Number of prototype for each classes
    :param num_clusters:      Number of clusters in feature dictionary
    :param prototype_file:    Path to the initial prototype weight (K-means clusters)
    :param feature_dict_file: Path to the initial feature dictionary weight (vMF clusters)
    :param recurrent_step:    Number of reccurence to filter out the occluded features

    :returns: TDMPNet model
    :rtype:   VGG_Attention_Prototype
    """
    return VGG_Attention_Prototype(
        Conv_Block,
        num_classes, num_prototype, num_clusters,
        prototype_file, feature_dict_file,
        prototype_method = prototype_method,
        recurrent_step = recurrent_step,
        percent_u = percent_u,
        percent_l = percent_l
    )


def initialize_vgg_attn(num_clusters:int, feature_dict_file:str, recurrent_step:int = 1, percent_u:float = 0.2, percent_l:float = 0.8):
    """
    Function to get original VGG + Attention model. Some hyperparameters are choosen in advance.

    :param num_clusters:      Number of clusters in feature dictionary
    :param feature_dict_file: Path to the initial feature dictionary weight (vMF clusters)
    :param recurrent_step:    Number of reccurence to filter out the occluded features

    :returns: VGG + Attention model
    :rtype:   VGG_Attention
    """
    return VGG_Attention(
        Conv_Block,
        num_clusters,
        feature_dict_file,
        recurrent_step = recurrent_step,
        percent_u = percent_u,
        percent_l = percent_l
    )


if __name__ == '__main__':
    root = '../dataset/MNIST_224X224_3'
    train_annot = 'pairs_train.txt'
    train_data = 'train'

    prototype_file = 'prototype_weight/original_vgg16.pkl'
    feature_dict_file = 'feature_dictionary/CASIA/dictionary_vgg_pool4_512.pickle'

    model = initialize_vgg_attn(128, feature_dict_file)
    summary(model, (3,224,224), device = 'cpu')

    # model.load_state_dict(torch.load('results/vgg16_attn_casia_3/epoch_1.pt'))