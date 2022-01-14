import torch
import torch.nn as nn


def cosine_sim(x1, x2, dim:int = 1, eps:float = 1e-8):
    """
    Compute Cosine Similarity of input x1 and x2.

    :param x1: First input tensor
    :type x1:   torch.FloatTensor

    :param x2: Second input tensor
    :type x2:   torch.FloatTensor

    :param dim: Dimension for L2 normalization
    :param eps: Lower threshold for clipping 
    """
    inner_product = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, p = 2, dim = dim)
    w2 = torch.norm(x2, p = 2, dim = dim)
    
    return inner_product / torch.outer(w1, w2).clamp(min = eps)


class MarginCosineProduct(nn.Module):
    """
    MCP layer implementation. This layer will substitute the fully connected layer
    (since face dataset has too many classes) and compute large margin cosine distance for each classes.
    """

    def __init__(self, in_features:int, out_features:int, s:float = 30.0, m:float = 0.40):
        """
        MCP layer initialization.

        :param in_features:  Number of input dimensions
        :param out_features: Number of output dimensions (number of classes)
        :param s:            Norm of input feature
        :param m:            Margin
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features # num class
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        """
        Forward propagation of MCP layer. This consists of:
           1. Compute cosine similarity of feature map with each class
           2. Convert the result on step 1 into one-hot representation
        
        :param x: Feature map of size (batch_size, in_features)
        :type x:   torch.FloatTensor

        :param label: Ground-truth label
        :type label:   torch.FloatTensor

        :returns: Tensor of output with shape (batch_size, num_classes)
        :rtype:   torch.FloatTensor 
        """
        cosine = cosine_sim(x, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter(1, label.reshape(-1,1), 1.0)

        output = self.s * (cosine - one_hot * self.m)

        return output