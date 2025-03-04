import os
import math
import torch
import torch.nn as nn
from  VU_BOIQA import VUBOIQA
from utils import train_one_epoch_IQA, test_IQA, compute_model, norm_loss_with_normalization, set_seed
from torch.utils.data import DataLoader
from MyDataset import MyDataset
#from Mydataset_OIQ import MyDataset
#from MyDataset_IQA import MyDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms as transforms
import warnings
from scipy.optimize import OptimizeWarning
import time
import sys
from config import EM360IQA_config
def main(cfg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(cfg)
    set_seed(42)
    model = VUBOIQA().to(cfg.device)
    checkpoint = torch.load('/mnt/10T/wkc/24_TOMM_VUBOIQA/VU-BIQA/VUIQA-JUFE/best_weight_VUIQA_Kad_10k_weitgh.pth', map_location='cuda:3')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.epochs)) / 2) * (1 - cfg.lrf) + cfg.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    loss_func = norm_loss_with_normalization       # norm-in-norm loss
    #loss_func = nn.L1Loss(reduction='mean')
    #loss_func = nn.MSELoss()
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    best_plcc = 0
    best_srcc = 0
    best_rmse = 0
    best_epoch = 0
    total_time = 0
    print("model:", cfg.model_name, "| dataset:", cfg.dataset_name, "| device:", cfg.device)
    for epoch in range(cfg.epochs):
        train_dataset = MyDataset(cfg.image_path,cfg.info_csv_path,'train',transform=train_transform,seed=epoch*10)
        test_dataset =  MyDataset(cfg.image_path,cfg.info_csv_path,'test',transform=test_transform,seed=epoch*10)
        train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=True,
    )
        test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
        # train
        start_time = time.time()
        train_loss, train_plcc, train_srcc, train_rmse= train_one_epoch_IQA(model, train_loader, loss_func, optimizer, epoch, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "[train epoch %d/%d] loss: %.6f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min, total time: %.2f h" % \
                (epoch+1, cfg.epochs, train_loss, train_plcc, train_srcc, train_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()
        
        scheduler.step()

        #test
        start_time = time.time()
        test_plcc, test_srcc, test_rmse = test_IQA(model, test_loader, epoch, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "[test  epoch %d/%d] PLCC: %.4f, SRCC: %.4f, RMSE: %.4f, LR: %.6f, TIME: %.2f MIN, total time: %.2f h" % \
                (epoch+1, cfg.epochs, test_plcc, test_srcc, test_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()

        if test_plcc + test_srcc > best_plcc + best_srcc:
            best_plcc = test_plcc
            best_srcc = test_srcc
            best_rmse = test_rmse
            best_epoch = epoch+1
            w_phat = cfg.save_ckpt_path + "/" + cfg.model_name + '-' + cfg.dataset_name
            if os.path.exists(w_phat) is False:
                os.makedirs(w_phat)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_plcc': best_plcc,
                'best_srcc': best_srcc,
                }, w_phat + "/best_weight_VUIQA_jufe_10k_pretrain.pth")
        
        if (epoch % 10 == 0 or epoch == (cfg.epochs-1)) and epoch > 0:
            print("="*80)
            print("[test epoch %d/%d] best_PLCC: %.4f, best_SRCC: %.4f, best_RMSE: %.4f" % (best_epoch, cfg.epochs, best_plcc, best_srcc, best_rmse))
            print("="*80)


if __name__ == '__main__': 
    cfg = EM360IQA_config()
    main(cfg)