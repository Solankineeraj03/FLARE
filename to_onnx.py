  
import argparse
import numpy as np

import os
import random
import logging
from tqdm import tqdm
from datetime import datetime
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator_40K, Discriminator_130K, Discriminator_6M


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return datetime.now().strftime('%m%d-%H')

def setup_logger(logger_name, expt, root, level=logging.INFO, screen=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, expt + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)



def evaluate_model(task_model, loader, device, criterion):

    task_model.eval()
    task_model = task_model.to(device)

    running_loss = 0.0
    running_corrects = 0

    total_labels = 0
    
    for data in loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = task_model(inputs)

        loss = criterion(logits, labels).item()

        _, preds = torch.max(logits.data, 1)

        running_loss += loss * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / total_labels
    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy


def infer_sampler(sampler, loader, device, logger):
    preds_all = []
    for data in loader:
        inputs, _ = data
        inputs = inputs.to(device)
        
        with torch.no_grad():
            preds = sampler(inputs)
            preds_all.extend(preds.squeeze().cpu().numpy())
    count_ranges(preds_all, logger)


import numpy as np

def count_ranges(values, logger):

    bins = np.arange(0, 1.1, 0.1)  # Bin edges: 0.0 to 1.0 inclusive
    hist, edges = np.histogram(values, bins=bins)

    result = {}
    for i in range(len(hist)):
        left = round(edges[i], 1)
        right = round(edges[i+1], 1)
        label = f"{left:.1f} - {right:.1f}"
        result[label] = hist[i]

    logger.info(f"Total numbers: {len(values)}")
    print(f"Total numbers: {len(values)}")
    for k, v in result.items():
        print(f"{k}: {v}")
        logger.info(f"{k}: {v}")
    
    return result

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', type=str, default='40k', help='Name of the experiment')
    parser.add_argument('--split', type=float, default=0.6, help='Number of epochs to train')
    parser.add_argument('--pretrain_epochs', type=int, default=4, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of data')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Name of dataset to use')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_images', type=int, default=50000, help='GPU id to use') 
    parser.add_argument('--num_val', type=int, default=5000, help='GPU id to use') 
    parser.add_argument('--budget', type=int, default=5000, help='GPU id to use') 
    parser.add_argument('--initial_budget', type=int, default=5000, help='GPU id to use') 
    parser.add_argument('--num_classes', type=int, default=10, help='GPU id to use') 
    parser.add_argument('--lr_task', type=float, default=0.01, help='Learning rate for compressed model') 
    parser.add_argument('--lr_disc', type=float, default=0.001, help='Learning rate for compressed model') 
    parser.add_argument('--lr_dec', type=float, default=0.01, help='Learning rate for decoder model') 
    parser.add_argument('--feat_wt', type=float, default=500, help='Weight for the auxilliary Feature loss') 
    parser.add_argument('--logit_wt', type=float, default=0.00005, help='Weight for the auxilliary Logit loss') 
    parser.add_argument('--adv_wt', type=float, default=0.1, help='Weight for the auxilliary Feature loss') 
    parser.add_argument('--seed', type=int, default=2024, help='Value for random seed') 
    parser.add_argument('--device', type=int, default=0, help='GPU id to use') 
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model') 
    parser.add_argument('--log_path', type=str, default='logs', help='Path to save logs') 

    args = parser.parse_args()


    set_random_seed(args.seed)

    mkdir(args.log_path)
    setup_logger('base', f'{args.expt}_{args.split}', args.log_path, level=logging.INFO)
    logger = logging.getLogger('base')

    logger.info(args)

    
    if args.save_path is None:
        save_path = f'checkpoints/'
    else:
        save_path = args.save_path
    
    mkdir(save_path)

    
    normalize = transforms.Normalize(
            mean= (0.5, 0.5, 0.5),
            std= (0.5, 0.5, 0.5), 
        )

    test_transform = transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.ToTensor(),
                    normalize,
            ])
    train_transform = transforms.Compose([
                    transforms.Resize([32, 32]),
                    # torchvision.transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ])
    

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False,
        download=True, transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=16, pin_memory=True
    )
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True,
        download=True, transform=train_transform
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, drop_last=True)
    


    fe_path = f'/home/rrgaire/projects/sampler_in_sensor/sampler_{args.expt}/ckpt_backup/CIFAR10_fe_model_{args.split}.pth'
    cl_path = f'/home/rrgaire/projects/sampler_in_sensor/sampler_{args.expt}/ckpt_backup/CIFAR10_classifier_model_{args.split}.pth'
    disc_path = f'/home/rrgaire/projects/sampler_in_sensor/sampler_{args.expt}/ckpt_backup/CIFAR10_sampler_model_{args.split}.pth'




    FE = CompVGGFeature()
    FE.load_state_dict(torch.load(fe_path))

    classifier = CompVGGClassifier(num_classes=args.num_classes)
    classifier.load_state_dict(torch.load(cl_path))

    if '40k' in args.expt or 'final' in args.expt:
        discriminator = Discriminator_40K()
    elif args.expt == '130k':
        discriminator = Discriminator_130K()
    discriminator.load_state_dict(torch.load(disc_path))

    task_model = nn.Sequential(FE, classifier)
    sampler = nn.Sequential(FE, discriminator)
    # sampler.feature_



    FE.eval()
    classifier.eval()
    discriminator.eval()

    FE = FE.to(args.device)
    classifier = classifier.to(args.device)
    discriminator = discriminator.to(args.device)

    eval_loss, eval_accuracy = evaluate_model(
        task_model=task_model,
        loader=test_loader,
        device=args.device,
        criterion=nn.CrossEntropyLoss()
        )   

    print(f'Task Accuracy: {eval_accuracy}')

    infer_sampler(sampler, train_dataloader, args.device, logger)
    infer_sampler(sampler, test_loader, args.device, logger)

    input = torch.randn(1, 3, 32, 32)
    input = input.to(args.device)

    # torch.onnx.export(
    # task_model,                  # model to export
    # (input,),        # inputs of the model,
    # f"{args.save_path}/task_model_{args.expt}_{args.split}.onnx",        # filename of the ONNX model
    # input_names=["input"],  # Rename inputs for the ONNX model
    # output_names=['output']
    # )   

    torch.save(sampler.state_dict(), os.path.join(save_path, f'sampler_{args.expt}_{args.split}.pth'))

    torch.onnx.export(
    sampler,                  # model to export
    (input,),        # inputs of the model,
    f"{save_path}/sampler_{args.expt}_{args.split}.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    output_names=['output']
    )

       
if __name__ == "__main__":
    main()
