#timm的基本使用
from ops_dcnv3.modules.dcnv3 import DCNv3_pytorch
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from torchvision import transforms
from thop import profile
from Mydataset_OIQ import MyDataset
#from my_dataset import MyDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math, copy
torch.cuda.empty_cache()
from torch import nn
from torchvision import models
from PIL import Image
backbone = nn.Sequential(*list(models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1).children())[:-1])
class VUBOIQA(nn.Module):
    def __init__(self):
        super(VUBOIQA,self).__init__()
        self.layer_1_1 = backbone[0][0]
        self.layer_1_2 = backbone[0][1]
        self.layer_1_3 = backbone[0][2]
        self.layer_2_1 = backbone[0][3]
        self.layer_2_2 = backbone[0][4]
        self.layer_3_1 = backbone[0][5]
        self.layer_3_2 = backbone[0][6]
        self.layer_4_1 = backbone[0][7]
        self.layer_finsh = backbone[1:]
        self.relu = nn.ReLU()
        self.adp = nn.AdaptiveAvgPool2d(1)
        self.dac_192 = DAA(192,384)
        self.dac_384 = DAA(384,768)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.DcnV3_96 = DCNv3_pytorch(96)
        self.DcnV3_192 = DCNv3_pytorch(192)
        self.DcnV3_384 = DCNv3_pytorch(384)
        self.DcnV3_768 = DCNv3_pytorch(768)
        self.Conv_1x1_96 = nn.Conv2d(96,192,kernel_size=1,stride=2)
        self.Conv_1x1_192 = nn.Conv2d(192,384,kernel_size=1,stride=2)
        self.Conv_1x1_384 = nn.Conv2d(384,768,kernel_size=1,stride=2)
        self.Conv_3x3_768 = nn.Conv2d(768,768,kernel_size=3,stride=1,padding=1)
        self.fusion  = FeatureFusion()
        self.epa_768 = HPA(channels=768)
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.patch_attention=PatchAttentionModule(channels=768,num_patches=10,embed_dim=768,num_heads=8,ffn_dim=1024,dropout=0.1)
        layers_to_freeze = [
            self.layer_1_1,
            self.layer_1_2,
            self.layer_1_3,
            self.layer_2_1,
            self.layer_2_2,
            self.layer_3_1,
            self.layer_3_2,
            self.layer_4_1
        ]
        for layer in layers_to_freeze:
                for param in layer.parameters():
                        param.requires_grad = False
    def fuse(self,x0,x1,x2,x3,x4):
        x0 = self.DcnV3_96(x0)
        x1 = self.DcnV3_192(x1)
        x2 = self.DcnV3_384(x2)
        x3 = self.DcnV3_768(x3)

        f0 = rearrange(x0,'b w h c -> b c h w')
        f1 = rearrange(x1,'b w h c -> b c h w')
        f2 = rearrange(x2,'b w h c -> b c h w')
        f3 = rearrange(x3,'b w h c -> b c h w')
        fuse_feature = self.Conv_1x1_96(f0)
        fuse_feature = rearrange(fuse_feature,'b c w h -> b w h c')
        fuse_feature = self.fusion(fuse_feature,x1)
        fuse_feature = self.dac_192(fuse_feature)
        fuse_l12 = self.Conv_1x1_192(f1) + f2
        fuse_l12 = rearrange(fuse_l12,'b w h c -> b c h w')
        fuse_l12 = self.DcnV3_384(fuse_l12)
        fuse_l23 = self.Conv_1x1_384(f2) + f3
        fuse_l23 = rearrange(fuse_l23,'b w h c -> b c h w')
        fuse_l23 = self.DcnV3_768(fuse_l23)
        fuse_feature = self.fusion(fuse_feature,fuse_l12)
        #DAC
        fuse_feature = self.dac_384(fuse_feature)
        fuse_feature = self.fusion(fuse_feature,fuse_l23)
        fuse_feature = self.DcnV3_768(fuse_feature)
        fuse_feature = rearrange(fuse_feature,'b w h c -> b c h w')
        fuse_feature = self.Conv_3x3_768(fuse_feature)
        fuse_feature = self.epa_768(fuse_feature)
        return fuse_feature
    def forward_vector(self,x):
        feature_vertor = []
        #x=x.permute(1,0,2,3,4)
        for i in range(x.shape[1]):
            input_x = x[:,i,:,:,:]
            #layer1
            out = self.layer_1_1(input_x)
            layer0_out = out
            out = self.layer_1_2(out)
            layer1_out = self.layer_1_3(out)
            #layer2
            out = self.layer_2_1(layer1_out)
            layer2_out = self.layer_2_2(out)
            #layer3
            out = self.layer_3_1(layer2_out)
            layer3_out = self.layer_3_2(out)
            #layer4
            out = self.layer_4_1(layer3_out)
            fuse_out = self.fuse(layer0_out,layer1_out,layer2_out,layer3_out,out)
            fuse_out = self.adp(fuse_out)
            fuse_out = self.flatten(fuse_out)
            #layer finish
            out = self.layer_finsh(out)
            feature_vertor.append(out+fuse_out)
        return feature_vertor
    def forward(self,x):
        feature_vertor = self.forward_vector(x)
        feature_vertor = rearrange(feature_vertor,'p b n -> b p n')
        feature_vertor = self.patch_attention(feature_vertor)
        out_put = self.fc1(feature_vertor)
        out_put = self.fc2(self.relu(self.dropout(out_put)))
        out_put = torch.squeeze(out_put,dim=-1)
        score = torch.mean(out_put,dim=1)
        return score
class DAA(nn.Module):
    def __init__(self, in_c, out_c):
        super(DAA, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dcn = DCNv3_pytorch(in_c)
        self.conv_1x1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_c, out_c // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_c // 4, out_c)
        self.sigmoid = nn.Sigmoid()
        self.conv_sa = nn.Conv2d(out_c, 1, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dcn(x)
        x = rearrange(x, 'b w h c -> b c h w')
        x = self.conv_1x1(x)
        w = self.global_avg_pool(x)  # [b, out_c, 1, 1]
        w = w.view(w.size(0), -1)    # [b, out_c]
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))  # [b, out_c]
        w = w.view(w.size(0), w.size(1), 1, 1)
        x_ca = x * w  # [b, out_c, h, w]

        w_sa = self.sigmoid(self.conv_sa(x_ca))  # [b, 1, h, w]
        x_sa = x_ca * w_sa  # [b, out_c, h, w]

        out = self.bn(x_sa)
        out = self.relu_final(out)

        out = F.max_pool2d(out, kernel_size=2, stride=2)

        out = rearrange(out, 'b c h w -> b w h c')
        return out
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1x1.bias, 0)
        nn.init.kaiming_normal_(self.conv_sa.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv_sa.bias, 0)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)

class HPA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(HPA, self).__init__()
        self.groups = factor
        print(channels)
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        weights = self.sigmoid(self.weight)
        return weights[0] * x1 + weights[1] * x2
class PatchAttentionModule(nn.Module):
    def __init__(self, channels, num_patches, embed_dim=256, num_heads=8, ffn_dim=512, dropout=0.1):
        super(PatchAttentionModule, self).__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.embedding(x)  
        x = x + self.pos_embedding  
        x = x.transpose(0, 1)  
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = x.transpose(0, 1)
        
        return x  

if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    net = VUBOIQA().to(device=device)
    x = torch.randn(1,10,3,224,224).to(device=device)
    flops, params = profile(net, (x,), verbose=False)
    print("FLOPs: %.1f G" % (flops / 1E9))
    print("Params: %.1f M" % (params / 1E6))
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = MyDataset('/mnt/10T/wkc/Database/JUFE_10k/final_dis_10320','/mnt/10T/wkc/Database/JUFE_10k/rjl.csv',
                              'train',transform=test_transform,seed=10)
    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=False,
    )
    for imgs,mos in test_loader:
        print(imgs.shape)
        out = net(imgs.to(device=device))
        print(out)
        print(mos)
        break
    