
import os
import torch
import copy
import time
import random
import logging
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_tsnee(split, train_dataset, current_indices, sampled_indices, FE):

    current_sampler = torch.utils.data.sampler.SubsetRandomSampler(current_indices)
    sampled_sampler = torch.utils.data.sampler.SubsetRandomSampler(sampled_indices)

    unlabeled_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=current_sampler, batch_size=1)
    sampled_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampled_sampler, batch_size=1)

    true_labels = []
    total_features = []

    for data in unlabeled_dataloader:
        inputs, labels, _ = data

        inputs = inputs.to(0)
        labels = labels.to(0)
        
        with torch.no_grad():
            features = FE(inputs)

        true_labels.append(labels.squeeze(0).cpu().numpy())
        total_features.append(inputs.flatten().squeeze(0).cpu().numpy())

    true_labels = np.array(true_labels)
    feature_array = np.array(total_features)
    print(true_labels.shape)
    print(feature_array.shape)

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(feature_array)

    	
    tsne = TSNE(n_components=2).fit_transform(pca_result_50)

    plot = sns.scatterplot(x = tsne[:, 0], y = tsne[:, 1], hue = true_labels, palette = sns.hls_palette(10), legend = 'full');

    fig = plot.get_figure()
    fig.savefig(f"{split}_current.png", dpi=1200, bbox_inches='tight') 


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




    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        return len(self.cifar10)
def cifar10_transformer():
    return torchvision.transforms.Compose([
            torchvision.transforms.Resize([32, 32]),
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                std=[0.2673, 0.2564, 0.2761]),
       ])
class CIFAR10(torch.utils.data.Dataset):
    def __init__(self, path):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=path,
            download=True,
            train=True,
            transform=cifar10_transformer()
        )
# def tiny_imagenet_transformer():
#     return torchvision.transforms.Compose([
#         torchvision.transforms.Resize(64),
#         torchvision.transforms.RandomCrop(64, padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         ),
#     ])

# class CIFAR100(torch.utils.data.Dataset):
#     def __init__(self, path):
#         self.cifar10 = torchvision.datasets.CIFAR100(root=path,
#                                         download=True,
#                                         train=True,
#                                         transform=cifar100_transformer()
#                                         )

#     def __getitem__(self, index):
#         if isinstance(index, np.float64):
#             index = index.astype(np.int64)

#         data, target = self.cifar10[index]

#         return data, target, index

#     def __len__(self):
#         return len(self.cifar10)

def load_weight(self, model, path):

    model.load_state_dict(torch.load(path))


def evaluate_model(FE, classifier, loader, device, criterion, state='val'):

    FE.eval()
    classifier.eval()
    FE.to(device)
    classifier.to(device)

    running_loss = 0.0
    running_corrects = 0

    total_labels = 0
    
    for data in loader:
        if state == 'val':
            inputs, labels, _ = data
        else:
            inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            features = FE(inputs)
            logits = classifier(features)

        loss = criterion(logits, labels).item()

        _, preds = torch.max(logits.data, 1)

        running_loss += loss * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / total_labels
    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy

def estimate_comm_cost(
    model, num_clients, num_rounds, bytes_per_param: int = 4
) -> dict:
    """
    Estimate total communication (in bytes) for federated rounds:
      - each client downloads + uploads one copy of the model per round.
    """
    num_params       = sum(p.numel() for p in model.parameters())
    model_size_bytes = num_params * bytes_per_param
    per_round_bytes  = num_clients * 2 * model_size_bytes
    total_bytes      = num_rounds * per_round_bytes
    return {
        "model_size_MB": model_size_bytes / 1e6,
        "per_round_MB":  per_round_bytes / 1e6,
        "total_comm_GB": total_bytes    / 1e9,
    }


def estimate_comm_cost_with_al(
    model,
    num_clients,
    num_rounds,
    bytes_per_param: int = 4,         # float32
    ble_throughput_bps: float = 1e6,  # 1 Mbps BLE
    budget_per_round: int = 100,      # e.g. you sample 100 indices
    bytes_per_index: int = 4,         # each index is a uint32
):
    # ----- Model‐exchange cost (as before) -----
    num_params       = sum(p.numel() for p in model.parameters())
    model_bytes      = num_params * bytes_per_param
    per_client_bytes = 2 * model_bytes
    model_tot_bytes  = num_rounds * num_clients * per_client_bytes

    # ----- AL “index” cost -----
    # each client sends `budget_per_round` indices each round
    al_per_client_bytes = budget_per_round * bytes_per_index
    al_tot_bytes        = num_rounds * num_clients * al_per_client_bytes

    # ----- Sum them up -----
    total_bytes = model_tot_bytes + al_tot_bytes

    # Convert to human units
    model_MB    = model_bytes      / 1e6
    per_round_MB= (num_clients * per_client_bytes) / 1e6
    al_per_round_MB = (num_clients * al_per_client_bytes) / 1e6
    total_GB    = total_bytes      / 1e9

    # BLE transfer times
    total_bits      = total_bytes  * 8
    time_total_s    = total_bits   / ble_throughput_bps

    return {
        "model_size_MB":     model_MB,
        "per_round_model_MB":per_round_MB,
        "per_round_al_MB":   al_per_round_MB,
        "total_comm_GB":     total_GB,
        "total_time_s":      time_total_s,
    }



def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def sample_for_labeling(args, budget, FE, decoder, discriminator, unlabeled_dataloader):
    all_preds = []
    all_indices = []

    FE.eval()
    discriminator.eval()
    decoder.eval()

    for images, _, indices in unlabeled_dataloader:
        images = images.to(args.device)

        with torch.no_grad():
            
            features = FE(images)

            preds = discriminator(features)

        preds = preds.cpu().data
        all_preds.extend(preds)
        all_indices.extend(indices)

    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1

    # select the points which the discriminator things are the most likely to be unlabeled
    _, querry_indices = torch.topk(all_preds, int(budget))
    querry_pool_indices = np.asarray(all_indices)[querry_indices]

    return querry_pool_indices

def pretrain_models(args, FE, classifier, querry_dataloader, val_dataloader, logger):

    ce_loss = nn.CrossEntropyLoss()

    optimizer_fe = torch.optim.SGD(FE.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)

    labeled_data = read_data(querry_dataloader, labels=True)

    train_iterations = (args.num_images) // args.batch_size


    FE.to(args.device) 
    classifier.to(args.device)   

    since = time.time()

    

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_fe_wt = copy.deepcopy(FE.state_dict())
    best_classifier_wt = copy.deepcopy(classifier.state_dict())

    for epoch in range(args.pretrain_epochs):

        running_ce_loss = 0.0
        running_corrects = 0

        FE.train()
        classifier.train()


        for iter in tqdm(range(train_iterations)):

            labeled_imgs, labels = next(labeled_data)

            labeled_imgs = labeled_imgs.to(args.device)
            labels = labels.to(args.device)


            optimizer_fe.zero_grad()
            optimizer_classifier.zero_grad()

            features = FE(labeled_imgs)

            logits = classifier(features)
           
            _, preds = torch.max(logits, 1)

            task_loss = ce_loss(logits, labels)


            task_loss.backward()
            optimizer_fe.step()
            optimizer_classifier.step()

            running_ce_loss += task_loss.item() * labeled_imgs.size(0)

            running_corrects += torch.sum(preds == labels.data)


        
        train_ce_loss = running_ce_loss / len(querry_dataloader.dataset)
        train_accuracy = running_corrects / len(querry_dataloader.dataset)


        val_loss, val_accuracy = evaluate_model(FE=FE,
                                                classifier=classifier,
                                                loader=val_dataloader,
                                                device=args.device,
                                                criterion=ce_loss)


        print(
                f'Pre-training Epoch: {epoch + 1} | {args.pretrain_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')
        logger.info(
                f'Pre-training Epoch: {epoch + 1} | {args.pretrain_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')


        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_fe_wt = copy.deepcopy(FE.state_dict())
            best_classifier_wt = copy.deepcopy(classifier.state_dict())

    time_elapsed = time.time() - since 

    print(f'Pre-Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc*100:4f}')
    print(f'Best train Acc: {best_train_acc*100:.4f}')

    logger.info(f'Pre-Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_val_acc*100:.4f}')
    logger.info(f'Best train Acc: {best_train_acc*100:.4f}')

    FE.load_state_dict(best_fe_wt)
    classifier.load_state_dict(best_classifier_wt)
        
    return FE, classifier

def infer_sampler(fe, disc, loader, device, logger):

    fe.eval()
    disc.eval()

    preds_all = []
    for data in loader:
        inputs, _, _ = data
        inputs = inputs.to(device)
        
        with torch.no_grad():
            preds = disc(fe(inputs)).squeeze()  # shape: [batch_size]
            preds_all.extend(preds.cpu().numpy())
    count_ranges(preds_all, logger)



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
    for k, v in result.items():
        logger.info(f"{k}: {v}")
    
    return result


def train_models(args, orig_model, FE, classifier, decoder, discriminator, querry_dataloader, val_dataloader, unlabeled_dataloader, logger):

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer_fe = torch.optim.SGD(FE.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)
    scheduler_fe = torch.optim.lr_scheduler.StepLR(optimizer_fe, step_size=150, gamma=0.1)

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)
    scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=150, gamma=0.1)

    optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr=args.lr_disc, weight_decay=5e-4, momentum=0.9)
    scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=150, gamma=0.1)

    optimizer_dec = torch.optim.SGD(decoder.parameters(), lr=args.lr_dec, weight_decay=5e-4, momentum=0.9)
    scheduler_dec = torch.optim.lr_scheduler.StepLR(optimizer_dec, step_size=150, gamma=0.1)

    labeled_data = read_data(querry_dataloader, labels=True)
    unlabeled_data = read_data(unlabeled_dataloader, labels=False)

    train_iterations = (args.num_images) // args.batch_size

    orig_model.to(args.device)
    FE.to(args.device)   
    classifier.to(args.device)  
    decoder.to(args.device)
    discriminator.to(args.device)   

    since = time.time()

    

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_weights = {}
    best_weights['FE'] = copy.deepcopy(FE.state_dict())
    best_weights['classifier'] = copy.deepcopy(classifier.state_dict())

    for epoch in range(args.train_epochs):

        running_ce_loss = 0.0
        running_bce_loss = 0.0
        running_feat_loss = 0.0
        running_logit_loss = 0.0
        running_corrects = 0

        orig_model.eval()
        FE.train()
        classifier.train()
        decoder.train()
        discriminator.train()


        for iter in tqdm(range(train_iterations)):

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(args.device)
            labels = labels.to(args.device)
            unlabeled_imgs = unlabeled_imgs.to(args.device)


            optimizer_fe.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_dec.zero_grad()

            lb_feat = FE(labeled_imgs)
            lb_logits = classifier(lb_feat)
           
            _, preds = torch.max(lb_logits, 1)

            classification_loss = ce_loss(lb_logits, labels)

            dec_lb_feats = decoder(lb_feat)

            with torch.no_grad():
                gt_feats, gt_logits = orig_model(labeled_imgs)

            feat_loss = mse_loss(dec_lb_feats, gt_feats)
            logit_loss = mse_loss(lb_logits, gt_logits)

            disc_lab = discriminator(lb_feat)

            disc_lab = disc_lab.squeeze(1)

            unlb_feat = FE(unlabeled_imgs)  # use this unlabelled features with orgianl model features and get a new loss. so that the compressed model learns unlabelld features as well.
            
            # dec_unlb_feats = decoder(unlb_feat)
            disc_unlab = discriminator(unlb_feat)
            disc_unlab = disc_unlab.squeeze(1)

            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
            lab_real_preds = lab_real_preds.to(args.device)
            unlab_real_preds = unlab_real_preds.to(args.device)

            # print(disc_unlab.shape, unlab_real_preds.shape)
            disc_loss = ( bce_loss(disc_lab, lab_real_preds) + bce_loss(disc_unlab, unlab_real_preds) ) / 2

            task_loss = classification_loss + args.adv_wt * disc_loss + args.feat_wt * feat_loss + args.logit_wt * logit_loss


            task_loss.backward()
            optimizer_classifier.step()
            optimizer_fe.step()
            optimizer_dec.step()


            # discriminator training

            optimizer_disc.zero_grad()

            with torch.no_grad():
                lb_feat = FE(labeled_imgs)
                unlb_feat = FE(unlabeled_imgs)
            
            labeled_preds = discriminator(lb_feat)
            unlabeled_preds = discriminator(unlb_feat)

            labeled_preds = labeled_preds.squeeze(1)
            unlabeled_preds = unlabeled_preds.squeeze(1)

            
            lab_real_preds = torch.ones(labeled_imgs.size(0))
            
            unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

            lab_real_preds = lab_real_preds.to(args.device)
            unlab_fake_preds = unlab_fake_preds.to(args.device)

            disc_loss = ( bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_fake_preds) ) / 2


            
            disc_loss.backward()
            optimizer_disc.step()


            

            running_ce_loss += classification_loss.item() * labeled_imgs.size(0)
            running_bce_loss += disc_loss.item() * labeled_imgs.size(0)
            running_feat_loss += feat_loss.item() * labeled_imgs.size(0)
            running_logit_loss += logit_loss.item() * labeled_imgs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        

        scheduler_fe.step()
        scheduler_classifier.step()
        scheduler_disc.step()
        scheduler_dec.step()
        
        train_ce_loss = running_ce_loss / len(querry_dataloader.dataset)
        train_bce_loss = running_bce_loss / len(querry_dataloader.dataset)
        train_feat_loss = running_feat_loss / len(querry_dataloader.dataset)
        train_logit_loss = running_logit_loss / len(querry_dataloader.dataset)
        train_accuracy = running_corrects / len(querry_dataloader.dataset)


        val_loss, val_accuracy = evaluate_model(FE=FE,
                                                classifier=classifier,
                                                loader=val_dataloader,
                                                device=args.device,
                                                criterion=ce_loss)


        print(
                f'Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train BCE Loss: {train_bce_loss:.8f}  Train FEAT Loss: {train_feat_loss:.8f} Train LOGIT Loss: {train_logit_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')
        logger.info(
                f'Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train BCE Loss: {train_bce_loss:.8f}  Train FEAT Loss: {train_feat_loss:.8f} Train LOGIT Loss: {train_logit_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')


        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_weights['FE'] = copy.deepcopy(FE.state_dict())
            best_weights['classifier'] = copy.deepcopy(classifier.state_dict())

    time_elapsed = time.time() - since 

    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc*100:4f}')
    print(f'Best train Acc: {best_train_acc*100:.4f}')

    logger.info(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_val_acc*100:.4f}')
    logger.info(f'Best train Acc: {best_train_acc*100:.4f}')
        
    return FE, classifier, decoder, discriminator, best_weights

                
           