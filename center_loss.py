import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    """
    def __init__(self, num_classes=1255, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim)).cuda() 
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            
        nn.init.normal_(self.centers, mean=0, std=1)  #tensor初始化，从而符合正态

    def forward(self, x, labels):

        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = torch.tensor(x, dtype=torch.float32).cuda()
        labels = labels.cuda()

        batch_size = x.size(0)   
        feature_size=x.size(1)
 
        
        #余弦相似度
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        x1=x.expand(self.num_classes,batch_size,feature_size).permute(1,0,2)  #[b,1255,256]
        center1=self.centers.expand(batch_size,self.num_classes,feature_size) #[b,1255,256]
        distmat=cos(x1,center1)  #[b,1255]余弦相似度
        distmat=torch.mul(distmat,-1)
        distmat=torch.add(distmat,1) #余弦距离

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)  #[,1255]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))  #[,1255]

        dist = distmat * mask.float()  #通过mask 找到中心点，并且不断减小样本与其对应类别中心之间的距离
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss