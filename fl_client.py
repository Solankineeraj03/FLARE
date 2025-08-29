# Its the newest one with 8 bit and msb lsb
# fl_client_msb8.py
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import flwr as fl

from models import (
    VGG,
    CompVGGFeature as FeatureExtractor,
    CompVGGClassifier as Classifier,
    Decoder,
    Discriminator,
)
from utils import cifar10_transformer, evaluate_model


# ---- helpers to make the Discriminator happy ----
def infer_disc_in_dim(discriminator: nn.Module, default: int = 64) -> int:
    """Infer in_features of the first Linear layer used by the Discriminator."""
    if hasattr(discriminator, "linear"):
        lin = discriminator.linear
        if isinstance(lin, nn.Linear):
            return lin.in_features
        if isinstance(lin, nn.Sequential) and len(lin) > 0:
            for m in lin:
                if isinstance(m, nn.Linear):
                    return m.in_features
    for m in discriminator.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return default


def to_d_vec(feats: torch.Tensor, target_c: int) -> torch.Tensor:
    """Convert [B,C,H,W] or [B,C] -> [B,target_c] by GAP + slice/pad."""
    if feats.dim() == 4:
        feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)  # [B,C]
    B, C = feats.shape
    if C > target_c:
        feats = feats[:, :target_c]
    elif C < target_c:
        pad = feats.new_zeros(B, target_c - C)
        feats = torch.cat([feats, pad], dim=1)
    return feats


# ---- 8-bit MSB/LSB controller (with per-batch apply) ----
class Int8MSBLSBController:
    """
    8-bit fake-quantization with two phases:
      - Phase 1 (epoch < switch_epoch): mask 2 LSBs (keep MSBs) -> gentler warmup
      - Phase 2 (epoch >= switch_epoch): full 8-bit
    Per-channel symmetric scale for tensors with dim >= 2; per-tensor otherwise.
    """

    def __init__(self, model: nn.Module, switch_epoch: int = 1, mask_bits: int = 2):
        self.model = model
        self.switch_epoch = switch_epoch
        self.mask_bits = int(mask_bits)
        self.prev_q8_msb = {}  # for warmup diagnostics

    @torch.no_grad()
    def _quantize_int8(self, w: torch.Tensor):
        # per-channel for tensors with dim >= 2
        if w.dim() >= 2:
            dims = list(range(1, w.dim()))
            max_abs = w.abs().amax(dim=dims, keepdim=True)
        else:
            max_abs = w.abs().max()
        scale = (max_abs / 127.0).clamp(min=1e-8)
        q = torch.round(w / scale).clamp_(-127, 127)  # float with integer values
        return q, scale

    @torch.no_grad()
    def _apply_mask_if_warmup(self, q8: torch.Tensor, local_epoch: int) -> torch.Tensor:
        if local_epoch < self.switch_epoch and self.mask_bits > 0:
            # zero out LSBs by right-shift then left-shift
            q32 = q8.to(torch.int32)
            q32 = (q32 >> self.mask_bits) << self.mask_bits
            q32 = q32.clamp(-128, 127)
            return q32.to(torch.int8).to(torch.float32)
        return q8  # float tensor with integer values

    @torch.no_grad()
    def apply(self, local_epoch: int):
        """Call once per epoch (for logging + epoch-level snap-to-grid)."""
        msb_drift = 0.0
        lsb_energy = 0.0

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            q8, scale = self._quantize_int8(p.data)

            if local_epoch < self.switch_epoch:
                # diagnostics in masked domain
                q8_msb = self._apply_mask_if_warmup(q8, local_epoch)
                if name in self.prev_q8_msb:
                    diff = (q8_msb.to(torch.int16) - self.prev_q8_msb[name].to(torch.int16)).float()
                    msb_drift += torch.norm(diff, p=2).item()
                self.prev_q8_msb[name] = q8_msb.clone().to(torch.int8)

                # measure LSB "energy" of original q8 (rough magnitude)
                q32 = q8.to(torch.int32)
                q8_lsb = (q32 & ((1 << self.mask_bits) - 1)).float()
                lsb_energy += torch.norm(q8_lsb, p=2).item()

                p.data.copy_(q8_msb * scale)
            else:
                p.data.copy_(q8 * scale)

        if local_epoch < self.switch_epoch:
            print(f"[LocalEpoch {local_epoch}] int8 MSB-drift={msb_drift:.4f} | LSB-energy={lsb_energy:.4f}")
        else:
            # phase-2: not reporting drift/energy
            print(f"[LocalEpoch {local_epoch}] (full 8-bit)")

    @torch.no_grad()
    def apply_per_batch(self, local_epoch: int):
        """Call after each optimizer step to keep weights on-grid."""
        for _, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            q8, scale = self._quantize_int8(p.data)
            q8 = self._apply_mask_if_warmup(q8, local_epoch)
            p.data.copy_(q8 * scale)


# ---- Flower client ----
class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
        self.cid = cid
        self.device = device
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.test_loader = test_loader
        self.args = args

        # Teacher (frozen)
        self.teacher = VGG(num_classes=args.num_classes).to(device)
        ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"),
                          map_location="cpu", weights_only=True)
        self.teacher.load_state_dict(ckpt, strict=True)
        self.teacher.eval()

        # Student + aux modules
        self.FE = FeatureExtractor().to(device)
        self.classifier = Classifier(num_classes=args.num_classes).to(device)
        self.decoder = Decoder().to(device)
        self.discriminator = Discriminator().to(device)

        # Discriminator input dim
        self.disc_in = infer_disc_in_dim(self.discriminator, default=64)

        # 8-bit MSB/LSB controller over (FE + Classifier)
        combined = nn.Sequential(self.FE, self.classifier)
        self.msb_lsb = Int8MSBLSBController(
            combined,
            switch_epoch=args.switch_epoch,
            mask_bits=2,  # 2 LSBs masked in warmup
        )

        # Losses
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bce_logits = nn.BCEWithLogitsLoss()  # stable GAN loss
        self.kd_T = args.kd_T

    # ---- federated params (FE + Classifier only) ----
    def get_parameters(self, config):
        params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
        params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
        return params

    def set_parameters(self, parameters):
        fe_len = len(list(self.FE.parameters()))
        fe_params = parameters[:fe_len]
        cl_params = parameters[fe_len:]
        for p, arr in zip(self.FE.parameters(), fe_params):
            p.data = torch.tensor(arr, device=self.device)
        for p, arr in zip(self.classifier.parameters(), cl_params):
            p.data = torch.tensor(arr, device=self.device)

    # ---- local train ----
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Windows-safe DataLoader
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        opt_student = optim.SGD(
            list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
            lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
        )
        opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

        for epoch in range(self.args.local_epochs):
            # epoch-level snap + (Phase-1) diagnostics
            self.msb_lsb.apply(epoch)

            self.teacher.eval()
            self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Teacher forward
                with torch.no_grad():
                    t_feats, t_logits = self.teacher(images)

                # Student forward
                s_feats = self.FE(images)
                s_logits = self.classifier(s_feats)
                dec_s_feats = self.decoder(s_feats)
                t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

                # Losses: task + distill (KL w/ temperature)
                cls_loss = self.ce(s_logits, labels)
                feat_loss = self.mse(dec_s_feats, t_feats_pooled)

                T = self.kd_T
                kd_loss = F.kl_div(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits.detach() / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)

                # Student adversarial (optional)
                adv_loss = 0.0
                if self.args.adv_wt > 0.0:
                    pred_fake_for_G = self.discriminator(to_d_vec(s_feats, self.disc_in)).squeeze(1)  # logits
                    adv_loss = self.bce_logits(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

                loss = cls_loss + self.args.distill_wt * (feat_loss + kd_loss) + self.args.adv_wt * adv_loss

                # Student step
                opt_student.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                opt_student.step()
                self.msb_lsb.apply_per_batch(epoch)  # keep weights on-grid every batch

                # Discriminator step (keep D in FP32)
                if self.args.adv_wt > 0.0:
                    with torch.no_grad():
                        s_feats_det = self.FE(images).detach()
                        t_feats_det, _ = self.teacher(images)
                    pred_real = self.discriminator(to_d_vec(t_feats_det, self.disc_in)).squeeze(1)  # logits
                    pred_fake = self.discriminator(to_d_vec(s_feats_det, self.disc_in)).squeeze(1)  # logits

                    loss_D = self.bce_logits(pred_real, torch.ones_like(pred_real)) + \
                             self.bce_logits(pred_fake, torch.zeros_like(pred_fake))
                    opt_disc.zero_grad()
                    loss_D.backward()
                    opt_disc.step()

        return self.get_parameters(config), len(self.train_dataset), {}

    # ---- local eval ----
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Windows-safe DataLoader for eval
        test_loader = DataLoader(
            self.test_loader.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        loss, acc = evaluate_model(
            FE=self.FE,
            classifier=self.classifier,
            loader=test_loader,
            device=self.device,
            criterion=self.ce,
            state='test',
        )
        return float(loss), len(test_loader.dataset), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_epochs", type=int, default=6)     # more full-8b epochs
    parser.add_argument("--lr_task", type=float, default=0.001)    # gentler for QAT
    parser.add_argument("--lr_disc", type=float, default=0.001)
    parser.add_argument("--distill_wt", type=float, default=1.0)
    parser.add_argument("--adv_wt", type=float, default=0.0)       # start with 0.0, re-enable later
    parser.add_argument("--kd_T", type=float, default=2.0)         # KD temperature
    parser.add_argument("--percent", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--index_dir", type=str, default="client_indices")
    parser.add_argument("--switch_epoch", type=int, default=1)      # enter full-8b early

    args = parser.parse_args()

    random.seed(args.cid)
    np.random.seed(args.cid)
    torch.manual_seed(args.cid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = cifar10_transformer()
    train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

    pct_i = int(100 * args.percent)
    lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
    train_ds = Subset(train_full, lab_idxs)

    # Build base test loader (dataset only; DataLoader rebuilt in evaluate())
    test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False, num_workers=0)

    client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
    # Note: Flower warns this is deprecated; still works. Their new "supernode" CLI is optional.
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())


if __name__ == "__main__":
    main()



# # fl_client_msb.py
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class MSBLSBController:
#     """
#     Two-phase coarse→fine training via uniform quantization:
#       Phase 1 (epoch < switch_epoch): clamp weights to coarse grid (step = 2^-msb_bits)
#       Phase 2: allow full precision updates (restore MSB + learn LSBs)
#     Notes:
#       - This is uniform quantization, not literal IEEE-754 bit-masking of MSBs.
#       - We track a cached 'MSB snapshot' per parameter for drift logging.
#     """
#     def __init__(self, model: nn.Module, msb_bits: int = 4, switch_epoch: int = 5):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def _quant_step(self):
#         # coarse grid step size
#         return 2.0 ** (-self.msb_bits)

#     def get_msb(self, weight: torch.Tensor) -> torch.Tensor:
#         step = self._quant_step()
#         return torch.round(weight / step) * step

#     @torch.no_grad()
#     def clamp_to_msb_only(self):
#         for name, p in self.model.named_parameters():
#             if not p.requires_grad:
#                 continue
#             msb = self.get_msb(p.data)
#             self.saved_msb[name] = msb.clone()
#             p.data.copy_(msb)

#     @torch.no_grad()
#     def restore_with_lsb_update(self):
#         # In this simplified uniform-quant version, we don't need to "restore" anything special:
#         # we just stop clamping and keep training in full precision. We keep saved MSB only for drift logs.
#         pass

#     @torch.no_grad()
#     def apply(self, local_epoch: int):
#         # Drift logging (optional diagnostics)
#         msb_drift = 0.0
#         lsb_norm = 0.0
#         for name, p in self.model.named_parameters():
#             if not p.requires_grad:
#                 continue
#             current = p.data
#             msb = self.get_msb(current)
#             if name in self.saved_msb:
#                 prev_msb = self.saved_msb[name]
#                 msb_drift += torch.norm(msb - prev_msb, p=2).item()
#             self.saved_msb[name] = msb.clone()
#             lsb = current - msb
#             lsb_norm += torch.norm(lsb, p=2).item()

#         print(f"[LocalEpoch {local_epoch}] MSB-drift={msb_drift:.4f} | LSB-norm={lsb_norm:.4f}")

#         # Phase control
#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"),
#                           map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + aux modules (decoder, discriminator not federated)
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # MSB/LSB controller over (FE + Classifier)
#         combined = nn.Sequential(self.FE, self.classifier)
#         self.msb_lsb = MSBLSBController(combined,
#                                         msb_bits=args.msb_bits,
#                                         switch_epoch=args.switch_epoch)

#         # Losses
#         self.ce = nn.CrossEntropyLoss()
#         self.mse = nn.MSELoss()
#         # Proper GAN losses with logits:
#         self.bce_logits = nn.BCEWithLogitsLoss()

#     # ---- federated param I/O (FE + Classifier only) ----
#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, arr in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)

#     # ---- local train ----
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for epoch in range(self.args.local_epochs):
#             # Apply coarse→fine control to FE+Classifier weights
#             self.msb_lsb.apply(epoch)

#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 # --- Teacher forward (features, logits) ---
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 # --- Student forward ---
#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)

#                 # Match teacher spatial dims to decoder output
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 # --- Losses: task + distill ---
#                 cls_loss = self.ce(s_logits, labels)
#                 feat_loss = self.mse(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse(s_logits, t_logits)

#                 # --- Adversarial (student tries to fool D as real=1) ---
#                 pred_fake_for_G = self.discriminator(s_feats).squeeze(1)     # logits
#                 adv_loss = self.bce_logits(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward()
#                 opt_student.step()

#                 # --- Discriminator step: real=teacher feats, fake=student feats ---
#                 with torch.no_grad():
#                     s_feats_det = self.FE(images).detach()
#                     t_feats_det, _ = self.teacher(images)  # teacher "real"
#                 # Optionally pool teacher feats to D's expected size or adapt D to accept both
#                 pred_real = self.discriminator(t_feats_det).squeeze(1)   # logits
#                 pred_fake = self.discriminator(s_feats_det).squeeze(1)   # logits

#                 loss_D = self.bce_logits(pred_real, torch.ones_like(pred_real)) + \
#                          self.bce_logits(pred_fake, torch.zeros_like(pred_fake))
#                 opt_disc.zero_grad()
#                 loss_D.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     # ---- local eval ----
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=4)

#     args = parser.parse_args()

#     # Seeds per client
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Data
#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False, num_workers=2)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# if __name__ == "__main__":
#     main()




# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=4, switch_epoch=10):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()

#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + AL modules
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # MSB-LSB controller (wrap both FE + Classifier)
#         combined_model = nn.Sequential(self.FE, self.classifier)
#         self.msb_lsb_controller = MSB_LSB_Controller(combined_model, args.msb_bits, args.switch_epoch)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().numpy() for p in self.FE.parameters()]
#         params += [p.detach().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for epoch in range(self.args.local_epochs):
#             self.msb_lsb_controller.apply_weight_logic(epoch)

#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_loss = (self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                              self.bce_loss(disc_real, torch.zeros_like(disc_real))) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=4)

#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl
# import zlib

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=6, switch_epoch=5):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()
#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#         self.msb_lsb_ctrl = MSB_LSB_Controller(
#             model=nn.Sequential(self.FE, self.classifier),
#             msb_bits=args.msb_bits,
#             switch_epoch=args.switch_epoch
#         )

#         self.current_round = 0

#     # In FedAvgClient class
#     def get_parameters(self, config):
#         msb_only = self.current_round < self.args.switch_epoch
#         all_params = list(self.FE.parameters()) + list(self.classifier.parameters())
#         tensors = []
#         for p in all_params:
#             if msb_only:
#                 arr = self.msb_lsb_ctrl.get_msb(p.data).cpu().numpy()
#             else:
#                 arr = (p.data - self.msb_lsb_ctrl.get_msb(p.data)).cpu().numpy()
#             tensors.append(arr)
#         return tensors

#     def set_parameters(self, parameters):
#         msb_only = self.current_round < self.args.switch_epoch
#         all_params = list(self.FE.parameters()) + list(self.classifier.parameters())
#         for p, arr in zip(all_params, parameters):
#             arr_tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
#             if msb_only:
#                 p.data = arr_tensor
#             else:
#                 msb = self.msb_lsb_ctrl.get_msb(p.data)
#                 p.data = msb + arr_tensor

#     def fit(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             self.msb_lsb_ctrl.apply_weight_logic(epoch)

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_loss = (self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                              self.bce_loss(disc_real, torch.zeros_like(disc_real))) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test'
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=20)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl
# from typing import List, Tuple, Dict

# from typing import List, Tuple, Dict
# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class MSB_LSB_Controller:
#     def __init__(self, model: nn.Module, msb_bits: int = 6):
#         self.model = model
#         self.msb_bits = msb_bits
#         # buffer previous integer repr per layer name
#         self.prev_int: Dict[str, torch.Tensor] = {}

#     def get_msb(self, weight: torch.Tensor) -> torch.Tensor:
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight: torch.Tensor, msb: torch.Tensor) -> torch.Tensor:
#         return weight - msb

#     def sparse_msb_delta(
#         self, name: str, weight: torch.Tensor
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Quantize weight → int16 MSB, diff against prev_int, return
#         (idxs: int32[], vals: int16[]).
#         """
#         step = 2 ** -self.msb_bits
#         msb = self.get_msb(weight)
#         # integer repr
#         cur_int = torch.round(msb / step).to(torch.int16)
#         # mask of changed positions
#         if name in self.prev_int:
#             mask = cur_int.ne(self.prev_int[name])
#         else:
#             mask = torch.ones_like(cur_int, dtype=torch.bool)
#         # update buffer
#         self.prev_int[name] = cur_int.clone()
#         # flatten & extract
#         flat_int = cur_int.view(-1).cpu().numpy()
#         flat_mask = mask.view(-1).cpu().numpy()
#         idxs = np.nonzero(flat_mask)[0].astype(np.int32)
#         vals = flat_int[flat_mask].astype(np.int16)
#         return idxs, vals


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(
#         self,
#         cid: int,
#         device: torch.device,
#         train_dataset,
#         unlabeled_dataset,
#         test_loader,
#         args,
#     ):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.mse_loss = nn.MSELoss()

#         # Quantizer / sparse‐delta controller
#         self.quantizer = MSB_LSB_Controller(
#             model=nn.Sequential(self.FE, self.classifier),
#             msb_bits=args.msb_bits,
#         )

#         self.current_round = 0

#     def get_parameters(self, config) -> List[np.ndarray]:
#         # Update current round
#         self.current_round = config.get("server_round", 0)
#         msb_only = self.current_round < self.args.switch_epoch

#         # Collect FE + classifier layers as (name, param)
#         layers = list(self.FE.named_parameters()) + list(self.classifier.named_parameters())
#         payload: List[np.ndarray] = []

#         for name, p in layers:
#             if msb_only:
#                 # sparse MSB‐delta
#                 idxs, vals = self.quantizer.sparse_msb_delta(name, p.data)
#                 payload.append(idxs)  # int32 array of indices
#                 payload.append(vals)  # int16 array of new values
#             else:
#                 # dense LSB residual
#                 msb = self.quantizer.get_msb(p.data)
#                 lsb = (p.data - msb).cpu().numpy().astype(np.float32)
#                 payload.append(lsb)

#         return payload

#     def set_parameters(self, parameters: List[np.ndarray]) -> None:
#         msb_only = self.current_round < self.args.switch_epoch
#         layers = list(self.FE.named_parameters()) + list(self.classifier.named_parameters())

#         i = 0
#         for name, p in layers:
#             if msb_only:
#                 first = parameters[i]
#                 if first.dtype.kind == "f":
#                      p.data.copy_(torch.tensor(first, dtype=torch.float32, device=self.device))
#                      i += 1
#                      continue
#                 # unpack sparse MSB‐delta

#                 idxs: np.ndarray = parameters[i]
#                 vals: np.ndarray = parameters[i + 1]
#                 i += 2

#                 # Ensure both are 1D arrays
#                 flat_idxs = idxs.ravel()  # Flatten to 1D
#                 flat_vals = vals.ravel()  # Flatten to 1D
            
#             # Validate shapes
#                 if flat_idxs.ndim != 1 or flat_vals.ndim != 1:
#                     raise ValueError(f"Expected 1D arrays, got idxs: {flat_idxs.shape}, vals: {flat_vals.shape}")
#                 if len(flat_idxs) != len(flat_vals):
#                     raise ValueError(f"Index/value mismatch: {len(flat_idxs)} vs {len(flat_vals)}")
            
#             # Reconstruct full int16 buffer
#                 total = p.data.numel()
#                 buf = torch.zeros(total, dtype=torch.int16, device="cpu")
            
#             # Convert to tensors
#                 idxs_tensor = torch.from_numpy(flat_idxs).long()
#                 vals_tensor = torch.from_numpy(flat_vals).short()
            
#             # Handle empty updates
#                 if len(idxs_tensor) > 0:
#                     buf.scatter_(0, idxs_tensor, vals_tensor)
            
#                 step = 2 ** -self.args.msb_bits
#                 msb = buf.to(torch.float32).view(p.data.shape) * step
#                 p.data.copy_(msb.to(self.device))

#             else:
#                 # dense LSB
#                 arr: np.ndarray = parameters[i]
#                 i += 1
#                 tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
#                 msb = self.quantizer.get_msb(p.data)
#                 p.data.copy_(msb + tensor)
#     def fit(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)

#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task,
#             momentum=0.9,
#             weight_decay=5e-4,
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc,
#             momentum=0.9,
#             weight_decay=5e-4,
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train()
#             self.decoder.train(); self.discriminator.train()

#             # (Optional) log MSB & LSB drift here
#             # self.quantizer.apply_weight_logic(epoch)

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 # Student forward
#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 # Losses
#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 # Discriminator step
#                 with torch.no_grad():
#                     s_feats2 = self.FE(images)
#                 disc_real = self.discriminator(s_feats2).squeeze(1)
#                 disc_loss = 0.5 * (
#                     self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                     self.bce_loss(disc_real, torch.zeros_like(disc_real))
#                 )
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         # Sync & unpack
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)

#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state="test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     test_full = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(
#         os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy")
#     )
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# -----------------------------------------------------------
#  AGZF-only Flower client (no MSB/LSB quantisation)
# -----------------------------------------------------------
# fl_client.py  —  AGZF-only Flower NumPyClient
# Updated fl_client.py with global-round AGZF freezing
# import argparse, os, random, numpy as np, torch
# import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision, flwr as fl
# from typing import List, Tuple

# from models import (
#     VGG,
#     CompVGGFeature as FE,
#     CompVGGClassifier as CL,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# # ------------------------------------------------------------
# # 1.  Adaptive Gradient-Zeroing Freezer (Global-Round-Based)
# # ------------------------------------------------------------
# class AGZF:
#     def __init__(self, modules: List[nn.Module], alpha=0.99, thr_w=1e-4, thr_g=1e-3, patience=3):
#         self.params = [p for m in modules for p in m.parameters()]
#         self.alpha, self.thr_w, self.thr_g, self.K = alpha, thr_w, thr_g, patience

#         self.ema_w = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in self.params]
#         self.ema_g = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in self.params]
#         self.prev_w = [p.detach().cpu().clone() for p in self.params]
#         self.counter = [torch.zeros_like(p, dtype=torch.int8, device="cpu") for p in self.params]
#         self.mask = [torch.ones_like(p, dtype=torch.bool, device="cpu") for p in self.params]

#     def zero_frozen_grads(self):
#         for p, m, ema_g in zip(self.params, self.mask, self.ema_g):
#             if p.grad is None:
#                 continue
#             g_cpu = p.grad.detach().cpu()
#             ema_g.mul_(self.alpha).add_(g_cpu.abs() * (1 - self.alpha))
#             g_cpu[~m] = 0.0
#             p.grad.copy_(g_cpu.to(p.device))

#     def update_masks(self):
#         froze_count, thawed_count = 0, 0
#         for i, (p, ema_w, ema_g, cnt, m, prev) in enumerate(zip(
#             self.params, self.ema_w, self.ema_g, self.counter, self.mask, self.prev_w
#         )):
#             w_cpu = p.detach().cpu()
#             delta = (w_cpu - prev).abs()
#             prev.copy_(w_cpu)
#             ema_w.mul_(self.alpha).add_(delta * (1 - self.alpha))

#             stable = (ema_w < self.thr_w) & (ema_g < self.thr_g)
#             cnt[stable] += 1
#             cnt[~stable] = 0

#             old_mask = m.clone()
#             m[cnt >= self.K] = False
#             m[cnt == 0] = True
#             froze_count += (old_mask & ~m).sum().item()
#             thawed_count += (~old_mask & m).sum().item()
#         return froze_count, thawed_count

#     def sync_prev_w(self):
#         for p, prev in zip(self.params, self.prev_w):
#             prev.copy_(p.detach().cpu())

#     def active_flat(self, layer_idx, tensor):
#         mask = self.mask[layer_idx].view(-1)
#         idxs = mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy().astype(np.int32)
#         vals = tensor.view(-1)[mask].detach().cpu().numpy().astype(np.float32)
#         return idxs, vals

#     def frozen_ratio(self):
#         frozen = sum((~m).sum().item() for m in self.mask)
#         total = sum(m.numel() for m in self.mask)
#         return frozen / total

# # ------------------------------------------------------------
# # 2.  Flower NumPyClient
# # ------------------------------------------------------------
# class Client(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, test_loader, args):
#         self.cid, self.device, self.args = cid, device, args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load("checkpoints/checkpoint.pth", map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True); self.teacher.eval()

#         self.FE, self.CL = FE().to(device), CL(num_classes=args.num_classes).to(device)
#         self.DEC, self.DISC = Decoder().to(device), Discriminator().to(device)
#         self.gzf = AGZF([self.FE, self.CL])

#         self.ce, self.mse = nn.CrossEntropyLoss(), nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.train_ds, self.test_loader = train_ds, test_loader

#     def get_parameters(self, _):
#         payload = []
#         for idx, p in enumerate(self.FE.parameters()):
#             payload += list(self.gzf.active_flat(idx, p))
#         offset = len(list(self.FE.parameters()))
#         for idx, p in enumerate(self.CL.parameters(), start=offset):
#             payload += list(self.gzf.active_flat(idx, p))
#         return payload

#     def set_parameters(self, _):
#         pass

#     def fit(self, parameters, config):
#         state = parameters if isinstance(parameters, list) else fl.common.parameters_to_ndarrays(parameters)
#         for p, arr in zip(list(self.FE.parameters()) + list(self.CL.parameters()), state):
#             p.data.copy_(torch.tensor(arr, device=self.device))

#         self.gzf.sync_prev_w()

#         loader = DataLoader(self.train_ds, batch_size=self.args.batch, shuffle=True, drop_last=True)
#         opt_s = optim.SGD(list(self.FE.parameters()) + list(self.CL.parameters()) + list(self.DEC.parameters()),
#                           lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
#         opt_d = optim.SGD(self.DISC.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

#         self.teacher.eval()

#         for epoch in range(self.args.local_epochs):
#             for batch_idx, (x, y) in enumerate(loader):
#                 x, y = x.to(self.device), y.to(self.device)
#                 with torch.no_grad():
#                     t_feat, t_log = self.teacher(x)
#                 s_feat = self.FE(x)
#                 s_log = self.CL(s_feat)
#                 dec = self.DEC(s_feat)
#                 t_pool = F.adaptive_avg_pool2d(t_feat, dec.shape[2:])

#                 loss = self.ce(s_log, y) + self.args.distill * (self.mse(dec, t_pool) + self.mse(s_log, t_log))
#                 opt_s.zero_grad(); loss.backward(retain_graph=True)
#                 self.gzf.zero_frozen_grads(); opt_s.step()

#                 disc = self.DISC(s_feat.detach()).squeeze(1)
#                 dloss = self.bce(disc, torch.ones_like(disc))
#                 opt_d.zero_grad(); dloss.backward(); opt_d.step()

#         froze, thawed = self.gzf.update_masks()
#         rnd = config.get("server_round", -1)
#         ratio = 100 * self.gzf.frozen_ratio()
#         print(f"[Client {self.cid}] Round {rnd:02d}: frozen {ratio:.1f}% of weights "+
#               f"(+{froze} froze, -{thawed} thawed this round)")

#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         state = parameters if isinstance(parameters, list) else fl.common.parameters_to_ndarrays(parameters)
#         for p, arr in zip(list(self.FE.parameters()) + list(self.CL.parameters()), state):
#             p.data.copy_(torch.tensor(arr, device=self.device))

#         loss, acc = evaluate_model(self.FE, self.CL, self.test_loader, self.device, self.ce, state="test")
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# # ------------------------------------------------------------
# # 3.  main()
# # ------------------------------------------------------------
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--cid", type=int, default=0)
#     p.add_argument("--batch", type=int, default=64)
#     p.add_argument("--lr", type=float, default=0.01)
#     p.add_argument("--local_epochs", type=int, default=1)
#     p.add_argument("--distill", type=float, default=1.0)
#     p.add_argument("--percent", type=float, default=1.0)
#     p.add_argument("--data_dir", type=str, default="data")
#     p.add_argument("--index_dir", type=str, default="client_indices")
#     p.add_argument("--num_classes", type=int, default=10)
#     args = p.parse_args()

#     random.seed(args.cid); np.random.seed(args.cid); torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     tf = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(args.data_dir, True, download=True, transform=tf)
#     test_full  = torchvision.datasets.CIFAR10(args.data_dir, False, download=True, transform=tf)

#     pct_i = int(args.percent * 100)
#     idx = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, idx)
#     test_loader = DataLoader(test_full, batch_size=args.batch, shuffle=False)

#     fl.client.start_numpy_client(
#         server_address="127.0.0.1:8080",
#         client=Client(args.cid, device, train_ds, test_loader, args),
#     )

# if __name__ == "__main__":
#     main()

# ------------------------------------------------------------
# ------------------------------------------------------------
# fl_client.py  –  Flower NumPyClient with AGZF + frozen-ratio log
# ------------------------------------------------------------
# import argparse, os, random, numpy as np, torch
# import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision, flwr as fl
# from typing import List, Tuple

# from models import (
#     VGG,
#     CompVGGFeature as FE,
#     CompVGGClassifier as CL,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model
# # ------------------------------------------------------------
# # 1.  Adaptive Gradient-Zeroing Freezer
# # ------------------------------------------------------------
# class AGZF:
#     """Dual-EMA per-parameter freezing with gradient zeroing."""

#     def __init__(
#         self,
#         modules: List[nn.Module],
#         alpha: float = 0.99,
#         thr_w: float = 1e-4,
#         thr_g: float = 1e-3,
#         patience: int = 3,
#     ):
#         self.params = [p for m in modules for p in m.parameters()]
#         self.alpha, self.thr_w, self.thr_g, self.K = alpha, thr_w, thr_g, patience

#         self.ema_w = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in self.params]
#         self.ema_g = [torch.zeros_like(p, dtype=torch.float32, device="cpu") for p in self.params]
#         self.prev_w = [p.detach().cpu().clone() for p in self.params]
#         self.counter = [torch.zeros_like(p, dtype=torch.int8, device="cpu") for p in self.params]
#         self.mask = [torch.ones_like(p, dtype=torch.bool, device="cpu") for p in self.params]

#     # ---------- hook 1: after backward -----------------------
#     def zero_frozen_grads(self) -> None:
#         for p, m, ema_g in zip(self.params, self.mask, self.ema_g):
#             if p.grad is None:
#                 continue
#             g_cpu = p.grad.detach().cpu()
#             ema_g.mul_(self.alpha).add_(g_cpu.abs() * (1 - self.alpha))
#             g_cpu[~m] = 0.0
#             p.grad.copy_(g_cpu.to(p.device))

#     # ---------- hook 2: after optimiser.step() ---------------
#     def update_masks(self) -> None:
#         for p, ema_w, ema_g, cnt, m, prev in zip(
#             self.params, self.ema_w, self.ema_g, self.counter, self.mask, self.prev_w
#         ):
#             w_cpu = p.detach().cpu()
#             delta = (w_cpu - prev).abs()
#             prev.copy_(w_cpu)

#             ema_w.mul_(self.alpha).add_(delta * (1 - self.alpha))

#             stable = (ema_w < self.thr_w) & (ema_g < self.thr_g)
#             cnt[stable] += 1
#             cnt[~stable] = 0

#             m[cnt >= self.K] = False   # freeze
#             m[cnt == 0] = True         # thaw

#     # ---------- helpers -------------------------------------
#     def active_flat(self, layer_idx: int, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
#         mask = self.mask[layer_idx].view(-1)
#         idxs = mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy().astype(np.int32)
#         vals = tensor.view(-1)[mask].detach().cpu().numpy().astype(np.float32)
#         return idxs, vals

#     def frozen_ratio(self) -> float:
#         frozen = sum((~m).sum().item() for m in self.mask)
#         total  = sum(m.numel()          for m in self.mask)
#         return frozen / total
# # ------------------------------------------------------------
# # 2.  Flower NumPyClient
# # ------------------------------------------------------------
# class Client(fl.client.NumPyClient):
#     def __init__(self, cid: int, device: torch.device, train_ds, test_loader, args):
#         self.cid, self.device, self.args = cid, device, args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load("checkpoints/checkpoint.pth", map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True); self.teacher.eval()

#         self.FE, self.CL = FE().to(device), CL(num_classes=args.num_classes).to(device)
#         self.DEC, self.DISC = Decoder().to(device), Discriminator().to(device)
#         self.gzf = AGZF([self.FE, self.CL])

#         self.ce, self.mse = nn.CrossEntropyLoss(), nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.train_ds, self.test_loader = train_ds, test_loader

#     # -------- Flower hooks --------
#     def get_parameters(self, _):
#         payload = []
#         for idx, p in enumerate(self.FE.parameters()):
#             payload += list(self.gzf.active_flat(idx, p))
#         offset = len(list(self.FE.parameters()))
#         for idx, p in enumerate(self.CL.parameters(), start=offset):
#             payload += list(self.gzf.active_flat(idx, p))
#         return payload

#     def set_parameters(self, _):  # server pushes full state in fit/evaluate
#         pass

#     # -------- training ----------
#     def fit(self, parameters, config):
#         state = parameters if isinstance(parameters, list) else fl.common.parameters_to_ndarrays(parameters)
#         for p, arr in zip(list(self.FE.parameters()) + list(self.CL.parameters()), state):
#             p.data.copy_(torch.tensor(arr, device=self.device))

#         loader = DataLoader(self.train_ds, batch_size=self.args.batch, shuffle=True, drop_last=True)
#         opt_s = optim.SGD(list(self.FE.parameters()) + list(self.CL.parameters()) + list(self.DEC.parameters()),
#                           lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
#         opt_d = optim.SGD(self.DISC.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
#         self.teacher.eval()

#         for epoch in range(self.args.local_epochs):
#             for batch_idx, (x, y) in enumerate(loader):
#                 x, y = x.to(self.device), y.to(self.device)
#                 with torch.no_grad():
#                     t_feat, t_log = self.teacher(x)

#                 s_feat = self.FE(x)
#                 s_log  = self.CL(s_feat)
#                 dec    = self.DEC(s_feat)
#                 t_pool = F.adaptive_avg_pool2d(t_feat, dec.shape[2:])

#                 loss = self.ce(s_log, y) + self.args.distill * (self.mse(dec, t_pool) + self.mse(s_log, t_log))
#                 opt_s.zero_grad(); loss.backward(retain_graph=True)
#                 self.gzf.zero_frozen_grads(); opt_s.step(); self.gzf.update_masks()

#                 disc = self.DISC(s_feat.detach()).squeeze(1)
#                 dloss = self.bce(disc, torch.ones_like(disc))
#                 opt_d.zero_grad(); dloss.backward(); opt_d.step()

#                 # -------- print frozen ratio once per global round --------
#                 if batch_idx == 0:   # first batch only
#                     rnd = config.get("server_round", -1)
#                     ratio = 100 * self.gzf.frozen_ratio()
#                     print(f"[Client {self.cid}] round {rnd:02d}  frozen {ratio:5.1f}% of weights")

#         return self.get_parameters(config), len(self.train_ds), {}

#     # -------- evaluation --------
#     def evaluate(self, parameters, config):
#         state = parameters if isinstance(parameters, list) else fl.common.parameters_to_ndarrays(parameters)
#         for p, arr in zip(list(self.FE.parameters()) + list(self.CL.parameters()), state):
#             p.data.copy_(torch.tensor(arr, device=self.device))

#         loss, acc = evaluate_model(self.FE, self.CL, self.test_loader, self.device, self.ce, state="test")
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# # ------------------------------------------------------------
# # 3.  main()
# # ------------------------------------------------------------
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--cid", type=int, default=0)
#     p.add_argument("--batch", type=int, default=64)
#     p.add_argument("--lr", type=float, default=0.01)
#     p.add_argument("--local_epochs", type=int, default=1)
#     p.add_argument("--distill", type=float, default=1.0)
#     p.add_argument("--percent", type=float, default=1.0)
#     p.add_argument("--data_dir", type=str, default="data")
#     p.add_argument("--index_dir", type=str, default="client_indices")
#     p.add_argument("--num_classes", type=int, default=10)
#     args = p.parse_args()

#     random.seed(args.cid); np.random.seed(args.cid); torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     tf = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(args.data_dir, True, download=True, transform=tf)
#     test_full  = torchvision.datasets.CIFAR10(args.data_dir, False, download=True, transform=tf)

#     pct_i = int(args.percent * 100)
#     idx = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, idx)
#     test_loader = DataLoader(test_full, batch_size=args.batch, shuffle=False)

#     fl.client.start_numpy_client(
#         server_address="127.0.0.1:8080",
#         client=Client(args.cid, device, train_ds, test_loader, args),
#     )

# if __name__ == "__main__":
#     main()


# this is the one i have to try
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl
# from typing import Dict, List, Tuple

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class MSB_LSB_Controller:
#     def __init__(self, model: nn.Module, msb_bits: int = 6):
#         self.model = model
#         self.msb_bits = msb_bits
#         # store previous 16-bit representation per layer
#         self.prev_int: Dict[str, torch.Tensor] = {}

#     def get_msb(self, weight: torch.Tensor) -> torch.Tensor:
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight: torch.Tensor, msb: torch.Tensor) -> torch.Tensor:
#         return weight - msb

#     def sparse_msb_delta(self, name: str, weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Quantize to MSB int16, diff against prev_int, return (idxs:int32[], vals:int16[]).
#         """
#         step = 2 ** -self.msb_bits
#         msb = self.get_msb(weight)
#         # integer representation
#         cur_int = torch.round(msb / step).to(torch.int16)
#         if name in self.prev_int:
#             changed_mask = cur_int.ne(self.prev_int[name])  # True where changed
#         else:
#             changed_mask = torch.ones_like(cur_int, dtype=torch.bool)
#         # update buffer
#         self.prev_int[name] = cur_int.clone()
#         # flatten and extract
#         flat_int = cur_int.view(-1).cpu().numpy()
#         flat_mask = changed_mask.view(-1).cpu().numpy()
#         idxs = np.nonzero(flat_mask)[0].astype(np.int32)
#         vals = flat_int[flat_mask].astype(np.int16)
#         return idxs, vals


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(
#         self,
#         cid: int,
#         device: torch.device,
#         train_dataset,
#         unlabeled_dataset,
#         test_loader,
#         args
#     ):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCEWithLogitsLoss()
#         self.mse_loss = nn.MSELoss()

#         # Quantizer
#         self.quantizer = MSB_LSB_Controller(
#             model=nn.Sequential(self.FE, self.classifier),
#             msb_bits=args.msb_bits
#         )
#         self.current_round = 0

#     def get_parameters(self, config):
#         """
#         Return either sparse MSB deltas (before switch_epoch) or dense LSB arrays.
#         """
#         self.current_round = config.get("server_round", 0)
#         msb_only = self.current_round < self.args.switch_epoch

#         params = list(nn.Sequential(self.FE, self.classifier).named_parameters())
#         payload: List[np.ndarray] = []
#         for name, p in params:
#             if msb_only:
#                 idxs, vals = self.quantizer.sparse_msb_delta(name, p.data)
#                 payload.append(idxs)   # int32 array
#                 payload.append(vals)   # int16 array
#             else:
#                 msb = self.quantizer.get_msb(p.data)
#                 lsb = self.quantizer.get_lsb(p.data, msb)
#                 payload.append(lsb.cpu().numpy().astype(np.float32))
#         return payload

#     def set_parameters(self, parameters):
#         """
#         In MSB-only phase: unpack sparse deltas to full MSB; 
#         Else: add full LSB.
#         """
#         params = list(nn.Sequential(self.FE, self.classifier).named_parameters())
#         msb_only = self.current_round < self.args.switch_epoch
#         i = 0
#         for name, p in params:
#             if msb_only:
#                 idxs = parameters[i]; vals = parameters[i+1]
#                 i += 2
#                 # reconstruct integer tensor
#                 total = p.data.numel()
#                 buf = torch.zeros(total, dtype=torch.int16)
#                 buf[idxs] = torch.from_numpy(vals)
#                 step = 2 ** -self.args.msb_bits
#                 msb = buf.to(torch.float32).view(p.data.shape) * step
#                 p.data.copy_(msb.to(self.device))
#             else:
#                 arr = parameters[i]; i += 1
#                 tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
#                 msb = self.quantizer.get_msb(p.data)
#                 p.data.copy_(msb + tensor)

#     def fit(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)

#         loader = DataLoader(
#             self.train_dataset,
#             batch_size=self.args.batch_size,
#             shuffle=True
#         )

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + 
#             list(self.classifier.parameters()) + 
#             list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train()
#             self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt*(feat_loss+logit_loss) + self.args.adv_wt*adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 with torch.no_grad():
#                     s_feats2 = self.FE(images)
#                 disc_real = self.discriminator(s_feats2).squeeze(1)
#                 disc_loss = 0.5*(self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                                  self.bce_loss(disc_real, torch.zeros_like(disc_real)))
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test'
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=5)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full  = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100*args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds  = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# current new msblsb
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl
# import zlib

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=6, switch_epoch=5):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()
#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#         self.msb_lsb_ctrl = MSB_LSB_Controller(
#             model=nn.Sequential(self.FE, self.classifier),
#             msb_bits=args.msb_bits,
#             switch_epoch=args.switch_epoch
#         )

#         self.current_round = 0

#     # In FedAvgClient class
#     def get_parameters(self, config):
#         msb_only = self.current_round < self.args.switch_epoch
#         all_params = list(self.FE.parameters()) + list(self.classifier.parameters())
#         tensors = []
#         for p in all_params:
#             if msb_only:
#                 arr = self.msb_lsb_ctrl.get_msb(p.data).cpu().numpy()
#             else:
#                 arr = (p.data - self.msb_lsb_ctrl.get_msb(p.data)).cpu().numpy()
#             tensors.append(arr)
#         return tensors

#     def set_parameters(self, parameters):
#         msb_only = self.current_round < self.args.switch_epoch
#         all_params = list(self.FE.parameters()) + list(self.classifier.parameters())
#         for p, arr in zip(all_params, parameters):
#             arr_tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
#             if msb_only:
#                 p.data = arr_tensor
#             else:
#                 msb = self.msb_lsb_ctrl.get_msb(p.data)
#                 p.data = msb + arr_tensor

#     def fit(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             self.msb_lsb_ctrl.apply_weight_logic(epoch)

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_loss = (self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                              self.bce_loss(disc_real, torch.zeros_like(disc_real))) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test'
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=5)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# this was good

# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=6, switch_epoch=5):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()
#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu")
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#         self.msb_lsb_ctrl = MSB_LSB_Controller(
#             model=nn.Sequential(self.FE, self.classifier),
#             msb_bits=args.msb_bits,
#             switch_epoch=args.switch_epoch
#         )

#         self.current_round = 0

#     def get_parameters(self, config):
#         fe_params = [p.cpu().detach().numpy() for p in self.FE.parameters()]
#         cl_params = [p.cpu().detach().numpy() for p in self.classifier.parameters()]
#         all_params = fe_params + cl_params
#         total_bytes = sum(p.nbytes for p in all_params)
#         print(f"[Client {self.cid}] Uploading {total_bytes / 1024:.2f} KB")
#         return all_params  # not wrapped with `ndarrays_to_parameters` here


#     # def set_parameters(self, parameters):
#     #     all_params = list(self.FE.parameters()) + list(self.classifier.parameters())
#     #     tensors = fl.common.parameters_to_ndarrays(parameters)

#     #     if self.current_round < self.args.switch_epoch:
#     #         for p, new in zip(all_params, tensors):
#     #             p.data = torch.tensor(new, dtype=torch.float16).to(self.device).float()
#     #     else:
#     #         for p, lsb in zip(all_params, tensors):
#     #             msb = self.msb_lsb_ctrl.get_msb(p.data)
#     #             p.data = msb + torch.tensor(lsb, dtype=torch.float16).to(self.device).float()
#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)


#     def fit(self, parameters, config):
#         self.current_round = config.get("server_round", 0)
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             self.msb_lsb_ctrl.apply_weight_logic(epoch)

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 adv_loss = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_loss = (self.bce_loss(disc_real, torch.ones_like(disc_real)) +
#                              self.bce_loss(disc_real, torch.zeros_like(disc_real))) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test'
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# #  fl_client.py fedavg without al also for fedadagrad
# import argparse
# import os
# import random
# import copy

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model
# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=4, switch_epoch=10):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()
#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         total_params = 0

#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()
#                 total_params += param.numel()

#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(
#         self, cid, device, train_dataset, unlabeled_dataset, test_loader, args
#     ):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + AL modules
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()
#         self.msb_lsb_ctrl = MSB_LSB_Controller(
#         model=nn.Sequential(self.FE, self.classifier),
#         msb_bits=args.msb_bits,
#         switch_epoch=args.switch_epoch
#         )
#     def get_parameters(self, config):
#         # Collect FE + Classifier weights only
#         params = [p.detach().numpy() for p in self.FE.parameters()]
#         params += [p.detach().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         # Split and load FE + Classifier params
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         # Load global weights
#         self.set_parameters(parameters)

#         # DataLoader
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         # Optimizer (FE + Classifier + Decoder + Discriminator)
#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         # Local training
#         for _ in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train(); self.msb_lsb_ctrl.apply_weight_logic(_)

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 # Teacher outputs
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 # Student forward
#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)

#                 # Pool teacher feats to student spatial dims
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 # Losses
#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 # Adversarial loss on features
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 real_labels = torch.ones_like(disc_pred)
#                 adv_loss = self.bce_loss(disc_pred, real_labels)

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 # Backward student + decoder
#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 # Train discriminator (real vs student features)
#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_fake_labels = torch.zeros_like(disc_real)
#                 disc_loss = (self.bce_loss(disc_real, real_labels) + self.bce_loss(disc_real, disc_fake_labels)) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         # Return updated FE + Classifier
#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=4)

#     args = parser.parse_args()

#     # Setup
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, Subset
# import torchvision
# import flwr as fl

# from models import ResNet18, DistillModel, Decoder, Discriminator
# from utils import cifar10_transformer, evaluate_model


# class SubsetWithIndices(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         real_idx = self.indices[i]
#         x, y = self.dataset[real_idx]
#         return x, y, real_idx

# class MSB_LSB_Controller:
#     def __init__(self, model, msb_bits=4, switch_epoch=10):
#         self.model = model
#         self.msb_bits = msb_bits
#         self.switch_epoch = switch_epoch
#         self.saved_msb = {}

#     def get_msb(self, weight):
#         step = 2 ** -self.msb_bits
#         return torch.round(weight / step) * step

#     def get_lsb(self, weight, msb):
#         return weight - msb

#     def clamp_to_msb_only(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 msb = self.get_msb(param.data)
#                 self.saved_msb[name] = msb.clone()
#                 param.data.copy_(msb)

#     def restore_with_lsb_update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and name in self.saved_msb:
#                 msb = self.saved_msb[name]
#                 lsb = self.get_lsb(param.data, msb)
#                 param.data.copy_(msb + lsb)

#     def apply_weight_logic(self, local_epoch):
#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()
#     def apply_weight_logic(self, local_epoch):
#         msb_drift = 0.0
#         lsb_drift = 0.0
#         total_params = 0

#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 current = param.data
#                 msb = self.get_msb(current)
#                 if name in self.saved_msb:
#                     prev_msb = self.saved_msb[name]
#                     msb_drift += torch.norm(msb - prev_msb, p=2).item()
#                 self.saved_msb[name] = msb.clone()
#                 lsb = self.get_lsb(current, msb)
#                 lsb_drift += torch.norm(lsb, p=2).item()
#                 total_params += param.numel()

#         print(f"[Epoch {local_epoch}] MSB Drift: {msb_drift:.4f} | LSB Norm: {lsb_drift:.4f}")

#         if local_epoch < self.switch_epoch:
#             self.clamp_to_msb_only()
#         else:
#             self.restore_with_lsb_update()

# class FedClientAL(fl.client.NumPyClient):
#     def __init__(self, cid, device, full_dataset, lab_idxs, unlab_idxs, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.args = args

#         self.train_ds = Subset(full_dataset, lab_idxs)
#         self.unlab_ds = SubsetWithIndices(full_dataset, unlab_idxs)
#         self.test_loader = test_loader

#         # === Load Teacher ===
#         backbone = ResNet18(num_classes=args.num_classes)
#         fe = nn.Sequential(
#             backbone.conv1, backbone.bn1, nn.ReLU(),
#             backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
#             nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
#         )
#         cl = backbone.linear

#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_resnet.pth",
#             map_location="cpu",
#             weights_only=True
#         )
#         fe.load_state_dict(ckpt["feature_extractor"])
#         cl.load_state_dict(ckpt["classifier"])
#         self.teacher = DistillModel(fe, cl).to(device)
#         self.teacher.eval()

#         # === Student ===
#         self.FE = fe.to(device)
#         self.classifier = nn.Linear(512, args.num_classes).to(device)
#         self.decoder = Decoder(in_features=512).to(self.device)  # for ResNet18 student

#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCELoss()
        # self.msb_lsb_ctrl = MSB_LSB_Controller(
        # model=nn.Sequential(self.FE, self.classifier),
        # msb_bits=args.msb_bits,
        # switch_epoch=args.switch_epoch
        # )


#     def get_parameters(self, config):
#         return [p.detach().cpu().numpy() for p in self.FE.parameters()] + \
#                [p.detach().cpu().numpy() for p in self.classifier.parameters()]

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         for p, arr in zip(self.FE.parameters(), parameters[:fe_len]):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), parameters[fe_len:]):
#             p.data = torch.tensor(arr, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)

#         lab_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds, batch_size=self.args.batch_size, shuffle=False)

#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         for _ in range(self.args.local_epochs):
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train(); self.msb_lsb_ctrl.apply_weight_logic(_)


#             for x, y in lab_loader:
#                 x, y = x.to(self.device), y.to(self.device)

#             # Teacher logits
#                 with torch.no_grad():
#                     t_logits, _ = self.teacher(x)

#             # Student forward pass
#                 s_feat = self.FE(x)
#                 s_logits = self.classifier(s_feat)
#                 s_recon = self.decoder(s_feat)

#             # Losses
#                 loss_cls = self.ce_loss(s_logits, y)
#                 loss_logit = self.mse_loss(s_logits, t_logits)  # match size [batch, num_classes]
#                 loss_recon = self.mse_loss(s_recon, x)  # assume decoder reconstructs the input image
#                 adv_score = self.discriminator(s_feat).squeeze(1)
#                 loss_adv = self.bce_loss(adv_score, torch.ones_like(adv_score))

#                 total_loss = loss_cls + self.args.distill_wt * (loss_logit + loss_recon) + self.args.adv_wt * loss_adv

#                 opt_student.zero_grad()
#                 total_loss.backward()
#                 opt_student.step()

#             # === Train Discriminator ===
#                 with torch.no_grad():
#                     feat_detach = self.FE(x).detach()
#                 disc_pred = self.discriminator(feat_detach).squeeze(1)
#                 loss_disc = self.bce_loss(disc_pred, torch.zeros_like(disc_pred))  # label as fake

#                 opt_disc.zero_grad()
#                 loss_disc.backward()
#                 opt_disc.step()

#         # === Active Sampling ===
#             self.FE.eval(); self.discriminator.eval()
#             all_scores, all_idxs = [], []

#             with torch.no_grad():
#                 for x_u, _, idx in unlab_loader:
#                     x_u = x_u.to(self.device)
#                     feats = self.FE(x_u)
#                     scores = self.discriminator(feats).squeeze(1).cpu()
#                     all_scores.append(scores)
#                     all_idxs.append(idx)

#             if all_scores:
#                 all_scores = torch.cat(all_scores)
#                 all_idxs = torch.cat(all_idxs)
#                 k = min(self.args.budget, len(all_scores))
#                 _, topk = torch.topk(-all_scores, k)  # most uncertain = lowest real/fake score
#                 picked = all_idxs[topk].tolist()

#                 self.train_ds.indices += picked
#                 self.unlab_ds.indices = [i for i in self.unlab_ds.indices if i not in picked]

#         return self.get_parameters(config), len(self.train_ds), {}


#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE, classifier=self.classifier, loader=self.test_loader,
#             device=self.device, criterion=self.ce_loss, state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=2)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--budget", type=int, default=100)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="./data")
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--teacher_ckpt", type=str, default="./checkpoint_resnet.pth")
# #     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--switch_epoch", type=int, default=10)
#     parser.add_argument("--msb_bits", type=int, default=4)

#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     full_train = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

#     pct = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct}.npy"))
#     unlab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct}.npy"))
    

#     client = FedClientAL(
#         cid=args.cid, device=device,
#         full_dataset=full_train,
#         lab_idxs=lab_idxs.tolist(),
#         unlab_idxs=unlab_idxs.tolist(),
#         test_loader=test_loader,
#         args=args
#     )
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# FLCLIENT FOR RESNET CIFAR-10
# fl_client.py for ResNet + CIFAR-10

# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import ResNet18, DistillModel
# from utils import cifar10_transformer, evaluate_model


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # === Load TEACHER (ResNet-based DistillModel) ===
#         backbone = ResNet18(num_classes=args.num_classes)

#         # Reconstruct feature_extractor
#         fe = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             nn.ReLU(),
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4,
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )
#         cl = backbone.linear

#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_resnet.pth",
#             map_location="cpu",
#             weights_only=True
#         )
#         fe.load_state_dict(ckpt["feature_extractor"])
#         cl.load_state_dict(ckpt["classifier"])
#         self.teacher = DistillModel(fe, cl).to(device)
#         self.teacher.eval()

#         # === STUDENT ===
#         self.FE = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             nn.ReLU(),
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4,
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         ).to(device)
#         self.classifier = nn.Linear(512, args.num_classes).to(device)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         optimizer = optim.SGD(
#             list(self.FE.parameters()) +
#             list(self.classifier.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train()
#             self.classifier.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_logits, _ = self.teacher(images)  # Only need logits

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 loss = cls_loss + self.args.distill_wt * logit_loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=20)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="./data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, test_loader, args)
#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=client.to_client(),
#     )

# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset, ConcatDataset
# import torchvision
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model


# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Initialize teacher model
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Initialize student components
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=100).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator(input_size=512*7*7).to(device)  # 512*7*7=25088

#         # Initialize losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCEWithLogitsLoss()  # More stable than BCELoss

#     def get_parameters(self, config):
#         # Only send FE and classifier parameters to server
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
        
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def active_sample(self, loader, budget=250):
#         self.FE.eval()
#         self.decoder.eval()
#         self.discriminator.eval()
#         scores = []

#         for images, _ in loader:
#             images = images.to(self.device)
#             feats = self.FE(images)
#             recon = self.decoder(feats)
#             disc_pred = self.discriminator(recon).view(-1)  # Raw logits
#             scores.extend(disc_pred.detach().cpu().tolist())

#         # Select samples with lowest discriminator scores
#         sorted_indices = np.argsort(scores)[:budget]
#         return sorted_indices

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)

#         # Active sampling from unlabeled pool
#         if len(self.unlabeled_dataset) > 0:
#             unlabeled_loader = DataLoader(self.unlabeled_dataset, batch_size=self.args.batch_size)
#             selected_indices = self.active_sample(unlabeled_loader, budget=250)
#             selected_subset = Subset(self.unlabeled_dataset, selected_indices)
#             self.train_dataset = ConcatDataset([self.train_dataset, selected_subset])

#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         # Optimizer for all components
#         optimizer = optim.SGD(
#             list(self.FE.parameters()) + 
#             list(self.classifier.parameters()) +
#             list(self.decoder.parameters()) + 
#             list(self.discriminator.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         # Training loop
#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train()
#             self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 # Teacher forward
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 # Student forward
#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 # Loss calculations
#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
                
#                 # Adversarial loss
#                 adv_logits = self.discriminator(dec_s_feats.detach()).view(-1)
#                 real_labels = torch.ones_like(adv_logits).to(self.device)
#                 adv_loss = self.bce_loss(adv_logits, real_labels)

#                 # Combined loss
#                 loss = (cls_loss + 
#                         self.args.distill_wt * (feat_loss + logit_loss) + 
#                         0.1 * adv_loss)

#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=3)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=100)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     # Set seeds for reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Prepare datasets
#     transform = cifar100_transformer()
#     train_full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     test_full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )

#     # Load client-specific indices
#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlabeled_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))

#     train_ds = Subset(train_full, lab_idxs)
#     unlabeled_ds = Subset(train_full, unlabeled_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     # Initialize and start client
#     client = FedAvgALClient(args.cid, device, train_ds, unlabeled_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset, ConcatDataset
# import torchvision
# import flwr as fl
# # At the top of your file:
# device = torch.device("cpu")



# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model


# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, unlabeled_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=100).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()
#         self.bce_loss = nn.BCELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def active_sample(self, loader, budget=250):
#         self.FE.eval()
#         self.decoder.eval()
#         self.discriminator.eval()
#         scores = []

#         for images, _ in loader:
#             images = images.to(self.device)
#             feats = self.FE(images)
#             recon = self.decoder(feats)
#             disc_pred = torch.sigmoid(self.discriminator(recon)).view(-1)
#             scores.extend(disc_pred.detach().cpu().tolist())

#         sorted_indices = np.argsort(scores)[:budget]
#         return sorted_indices

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)

#         if len(self.unlabeled_dataset) > 0:
#             unlabeled_loader = DataLoader(self.unlabeled_dataset, batch_size=self.args.batch_size)
#             selected_indices = self.active_sample(unlabeled_loader, budget=250)
#             selected_subset = Subset(self.unlabeled_dataset, selected_indices)
#             self.train_dataset = ConcatDataset([self.train_dataset, selected_subset])

#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         optimizer = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) +
#             list(self.decoder.parameters()) + list(self.discriminator.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 adv_logits = self.discriminator(dec_s_feats.detach())
#                 real_labels = torch.ones_like(adv_logits).to(self.device)
#                 adv_loss = self.bce_loss(torch.sigmoid(adv_logits), real_labels)

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + 0.1 * adv_loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     # Then in your main block:
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=3)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=100)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cpu")

#     transform = cifar100_transformer()
#     train_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlabeled_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))

#     train_ds = Subset(train_full, lab_idxs)
#     unlabeled_ds = Subset(train_full, unlabeled_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgALClient(args.cid, device, train_ds, unlabeled_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# import argparse
# import os
# import random
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl
# from itertools import cycle
# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model

# def set_random_seed(seed: int = 42):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(
#         self,
#         cid: int,
#         device: torch.device,
#         train_dataset: Subset,
#         unlabeled_dataset: Subset,
#         test_loader: DataLoader,
#         args,
#     ):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher network (frozen)
#         self.orig_model = VGG(num_classes=100).to(device)
        
#         # Student networks
#         self.FE = FeatureExtractor().to(device)           # Federated
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)  # Federated
#         self.decoder = Decoder().to(device)               # Local only
#         self.discriminator = Discriminator().to(device)   # Federated

#         # Load teacher checkpoint
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Precompute parameter counts for federated models
#         self.num_fe_params = len(list(self.FE.parameters()))
#         self.num_cl_params = len(list(self.classifier.parameters()))
#         self.num_disc_params = len(list(self.discriminator.parameters()))

#         # Loss functions
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         # Return parameters for federated models only (FE, classifier, discriminator)
#         params = [p.cpu().detach().numpy() for p in self.FE.parameters()]
#         params += [p.cpu().detach().numpy() for p in self.classifier.parameters()]
#         # params += [p.cpu().detach().numpy() for p in self.discriminator.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         # Split parameters into federated components
#         fe_params = parameters[:self.num_fe_params]
#         cl_params = parameters[self.num_fe_params:self.num_fe_params+self.num_cl_params]
#         # disc_params = parameters[self.num_fe_params+self.num_cl_params:self.num_fe_params+self.num_cl_params+self.num_disc_params]
        
#         # Set FE parameters
#         for p, new_val in zip(self.FE.parameters(), fe_params):
#             if p.data.shape == new_val.shape:
#                 p.data = torch.tensor(new_val, device=self.device)
        
#         # Set classifier parameters
#         for p, new_val in zip(self.classifier.parameters(), cl_params):
#             if p.data.shape == new_val.shape:
#                 p.data = torch.tensor(new_val, device=self.device)
        
        

#     def fit(self, parameters, config):
#         # 1) Load global weights for federated models
#         self.set_parameters(parameters)
        
#         # 2) Build dataloaders
#         querry_loader = DataLoader(
#             self.train_dataset, batch_size=self.args.batch_size, shuffle=True
#         )
#         unlabeled_loader = DataLoader(
#             self.unlabeled_dataset, batch_size=self.args.batch_size, shuffle=False
#         )
        
#         # 3) Optimizers - separate for federated and local models
#         # Federated models optimizer
#         opt_fed = optim.SGD(
#             list(self.FE.parameters()) + 
#             list(self.classifier.parameters()) + 
#             list(self.discriminator.parameters()),
#             lr=self.args.lr_task, 
#             momentum=0.9, 
#             weight_decay=5e-4
#         )
#         sch_fed = optim.lr_scheduler.StepLR(opt_fed, step_size=150, gamma=0.1)
        
#         # Local decoder optimizer
#         opt_dec = optim.SGD(
#             self.decoder.parameters(), 
#             lr=self.args.lr_dec, 
#             momentum=0.9, 
#             weight_decay=5e-4
#         )
#         sch_dec = optim.lr_scheduler.StepLR(opt_dec, step_size=150, gamma=0.1)
        
#         # 4) Round-robin iterators
#         labeled_iter = cycle(querry_loader)
#         if len(self.unlabeled_dataset) > 0:
#             unlabeled_iter = cycle(unlabeled_loader)
#         else:
#             unlabeled_iter = None

#         iterations = len(self.train_dataset) // self.args.batch_size
#         best_val_acc = 0.0
#         best_weights = {
#             "FE": copy.deepcopy(self.FE.state_dict()),
#             "classifier": copy.deepcopy(self.classifier.state_dict()),
#             "discriminator": copy.deepcopy(self.discriminator.state_dict()),
#         }

#         # 5) Local training
#         for epoch in range(self.args.train_epochs):
#             self.orig_model.eval()
#             self.FE.train(); self.classifier.train();
#             self.decoder.train(); self.discriminator.train()

#             for _ in range(iterations):
#                 lb_imgs, labels = next(labeled_iter)
#                 lb_imgs, labels = lb_imgs.to(self.device), labels.to(self.device)

#                 # attempt unlabeled batch
#                 if unlabeled_iter is not None:
#                     try:
#                         unlb_imgs, _ = next(unlabeled_iter)
#                         unlb_imgs = unlb_imgs.to(self.device)
#                     except StopIteration:
#                         unlb_imgs = None
#                 else:
#                     unlb_imgs = None

#                 # Zero out gradients
#                 opt_fed.zero_grad()
#                 opt_dec.zero_grad()

#                 # (a) Student forward
#                 lb_feat = self.FE(lb_imgs)
#                 lb_logits = self.classifier(lb_feat)
#                 classification_loss = self.ce_loss(lb_logits, labels)

#                 # (b) Decoder + mimic losses (local only)
#                 dec_lb = self.decoder(lb_feat)
#                 with torch.no_grad():
#                     gt_feat, gt_logits = self.orig_model(lb_imgs)
#                 feat_loss = self.mse_loss(dec_lb, gt_feat)
#                 logit_loss = self.mse_loss(lb_logits, gt_logits)

#                 # (c) Adversarial loss (federated)
#                 disc_lab = self.discriminator(lb_feat).squeeze(1)
#                 lab_real = torch.ones_like(disc_lab, device=self.device)
#                 if unlb_imgs is not None:
#                     unlb_feat = self.FE(unlb_imgs)
#                     disc_unlab = self.discriminator(unlb_feat).squeeze(1)
#                     unlab_real = torch.ones_like(disc_unlab, device=self.device)
#                     disc_loss = (self.bce_loss(disc_lab, lab_real) +
#                                  self.bce_loss(disc_unlab, unlab_real)) / 2
#                 else:
#                     disc_loss = self.bce_loss(disc_lab, lab_real)

#                 # (d) Total losses
#                 fed_loss = (classification_loss
#                             + self.args.adv_wt * disc_loss
#                             + self.args.logit_wt * logit_loss)
                
#                 local_loss = self.args.feat_wt * feat_loss
                
#                 # Backpropagate
#                 fed_loss.backward(retain_graph=True)
#                 local_loss.backward()
                
#                 # Update parameters
#                 opt_fed.step()
#                 opt_dec.step()

#             sch_fed.step()
#             sch_dec.step()

#             # Evaluate on the query (labeled) split
#             val_loss, val_acc = evaluate_model(
#                 FE=self.FE,
#                 classifier=self.classifier,
#                 loader=querry_loader,
#                 device=self.device,
#                 criterion=self.ce_loss,
#                 state='test',
#             )
#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 best_weights['FE'] = copy.deepcopy(self.FE.state_dict())
#                 best_weights['classifier'] = copy.deepcopy(self.classifier.state_dict())
#                 best_weights['discriminator'] = copy.deepcopy(self.discriminator.state_dict())

#             print(f"Epoch {epoch+1}/{self.args.train_epochs} | Val Acc: {val_acc*100:.2f}%")

#         # Restore best weights
#         self.FE.load_state_dict(best_weights['FE'])
#         self.classifier.load_state_dict(best_weights['classifier'])
#         self.discriminator.load_state_dict(best_weights['discriminator'])

#         # Active learning sampling
#         if len(self.unlabeled_dataset) > 0:
#             all_feats = []
#             for imgs, _ in unlabeled_loader:
#                 imgs = imgs.to(self.device)
#                 with torch.no_grad():
#                     all_feats.append(self.FE(imgs).cpu())
#             all_feats = torch.cat(all_feats, 0)
#             scores = self.discriminator(all_feats.to(self.device))
#             scores = scores.detach().cpu().numpy().flatten()
            
#             # Select samples with lowest discriminator scores
#             # k = min(self.args.budget, scores.size)
#             # selected = np.argpartition(scores, k)[:k]
#             k = min(self.args.budget, scores.size)
#             if k == scores.size:
#                 selected = np.arange(k)
#             else:
#                 selected = np.argpartition(scores, k)[:k]

            
#             # Update datasets
#             current = list(self.train_dataset.indices)
#             new_samples = [self.unlabeled_dataset.indices[i] for i in selected]
#             self.train_dataset.indices = current + new_samples
            
#             # Remove selected samples from unlabeled data
#             mask = np.ones(len(self.unlabeled_dataset.indices), dtype=bool)
#             mask[selected] = False
#             self.unlabeled_dataset.indices = self.unlabeled_dataset.indices[mask]
            
#             print(f"[Client {self.cid}] Added {len(new_samples)} samples; total = {len(self.train_dataset.indices)}")
#         else:
#             print(f"[Client {self.cid}] No unlabeled data left to sample")

#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=1)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=100)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_dec", type=float, default=0.001)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--train_epochs", type=int, default=5)
#     parser.add_argument("--budget", type=int, default=500)
#     parser.add_argument("--adv_wt", type=float, default=1.0)
#     parser.add_argument("--feat_wt", type=float, default=500.0)
#     parser.add_argument("--logit_wt", type=float, default=0.00005)
#     parser.add_argument(
#         "--percent", type=float, default=1.0, help="Fraction of data for this client split"
#     )
#     parser.add_argument(
#         "--index_dir", type=str, default="client_indices", help="Directory with split index .npy files"
#     )
#     parser.add_argument("--device", type=int, default=0, help="GPU device index")
#     args = parser.parse_args()

#     set_random_seed(args.cid)
#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
#     transform = cifar100_transformer()

#     train_full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     test_data = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

#     # Load splits
#     split_pct = int(args.percent * 100)
#     base = os.path.dirname(os.path.abspath(__file__))
#     lab_file = os.path.join(base, args.index_dir, f"client_{args.cid}_split_{split_pct}.npy")
#     unlab_file = os.path.join(base, args.index_dir, f"client_{args.cid}_unlabeled_split_{split_pct}.npy")
#     labeled_indices = np.load(lab_file)
#     unlabeled_indices = np.load(unlab_file)

#     train_dataset = Subset(train_full, labeled_indices)
#     unlabeled_dataset = Subset(train_full, unlabeled_indices)

#     client = FedAvgALClient(args.cid, device, train_dataset, unlabeled_dataset, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# # fl_client.py VGG 
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from utils import tiny_imagenet_transformer
# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
# )
# from utils import evaluate_model


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"/home/nsola5/Sampler/checkpoints/checkpoint_tinyimagenet.pth",
#             map_location="cpu",
#             weights_only=True,
#         )

#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=200).to(device)
        
#         self.decoder = Decoder().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         optimizer = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 print(s_logits[0])
#                 dec_s_feats = self.decoder(s_feats)

#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     # def evaluate(self, parameters, config):
#     #     self.set_parameters(parameters)
#     #     loss, acc, logits = evaluate_model(
#     #         FE=self.FE,
#     #         classifier=self.classifier,
#     #         loader=self.test_loader,
#     #         device=self.device,
#     #         criterion=self.ce_loss,
#     #         state='test',
#     #     )
#     #     print("logits shape:", logits.shape)  # Should be [batch_size, 100]

#     #     return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}
#     # fl_client.py
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
    
#     # Only get loss and accuracy (no logits)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )

#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=20)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="C:\\Users\\nsola5\\OneDrive - University of Illinois Chicago\\Desktop\\GAnn\\tiny-imagenet-200\\tiny-imagenet-200")

#     parser.add_argument("--num_classes", type=int, default=200)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")

#     # transform = cifar100_transformer()
#     # train_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
#     # test_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
   
#     transform = tiny_imagenet_transformer()
#     train_full = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir, "train"), transform=transform)
#     test_full = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir, "val_fixed"), transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="131.193.50.219:5000", client=client)

    
# fl_client.py
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
# )
# from utils import cifar100_transformer, evaluate_model


# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_dataset, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"/home/nsola5/Sampler/checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )

#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=100).to(device)
        
#         self.decoder = Decoder().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.discriminator.parameters()]
#         return params


#     def set_parameters(self, parameters):
#         fe_len  = len(list(self.FE.parameters()))
#         cl_len  = len(list(self.classifier.parameters()))
#         disc_len = len(list(self.discriminator.parameters()))

#         fe_params   = parameters[:fe_len]
#         cl_params   = parameters[fe_len:fe_len + cl_len]
#         disc_params = parameters[fe_len + cl_len:]

#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.discriminator.parameters(), disc_params):
#             p.data = torch.tensor(new, device=self.device)


#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         optimizer = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 print(s_logits[0])
#                 dec_s_feats = self.decoder(s_feats)

#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         return self.get_parameters(config), len(self.train_dataset), {}

#     # def evaluate(self, parameters, config):
#     #     self.set_parameters(parameters)
#     #     loss, acc, logits = evaluate_model(
#     #         FE=self.FE,
#     #         classifier=self.classifier,
#     #         loader=self.test_loader,
#     #         device=self.device,
#     #         criterion=self.ce_loss,
#     #         state='test',
#     #     )
#     #     print("logits shape:", logits.shape)  # Should be [batch_size, 100]

#     #     return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}
#     # fl_client.py
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
    
#     # Only get loss and accuracy (no logits)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )

#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=10)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=100)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cuda:0")

#     transform = cifar100_transformer()
#     train_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
   


#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# # fl_client.py fedavg without al also for fedadagrad
# import argparse
# import os
# import random
# import copy

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class FedAvgClient(fl.client.NumPyClient):
#     def __init__(
#         self, cid, device, train_dataset, unlabeled_dataset, test_loader, args
#     ):
#         self.cid = cid
#         self.device = device
#         self.train_dataset = train_dataset
#         self.unlabeled_dataset = unlabeled_dataset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + AL modules
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         # Collect FE + Classifier weights only
#         params = [p.detach().numpy() for p in self.FE.parameters()]
#         params += [p.detach().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         # Split and load FE + Classifier params
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         # Load global weights
#         self.set_parameters(parameters)

#         # DataLoader
#         loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

#         # Optimizer (FE + Classifier + Decoder + Discriminator)
#         opt_student = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(self.discriminator.parameters(), lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4)

#         # Local training
#         for epoch in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train(); self.decoder.train(); self.discriminator.train()

#             for images, labels in loader:
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 # Teacher outputs
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(images)

#                 # Student forward
#                 s_feats = self.FE(images)
#                 s_logits = self.classifier(s_feats)
#                 dec_s_feats = self.decoder(s_feats)

#                 # Pool teacher feats to student spatial dims
#                 t_feats_pooled = F.adaptive_avg_pool2d(t_feats, dec_s_feats.shape[2:])

#                 # Losses
#                 cls_loss = self.ce_loss(s_logits, labels)
#                 feat_loss = self.mse_loss(dec_s_feats, t_feats_pooled)
#                 logit_loss = self.mse_loss(s_logits, t_logits)
#                 # Adversarial loss on features
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 real_labels = torch.ones_like(disc_pred)
#                 adv_loss = self.bce_loss(disc_pred, real_labels)

#                 loss = cls_loss + self.args.distill_wt * (feat_loss + logit_loss) + self.args.adv_wt * adv_loss

#                 # Backward student + decoder
#                 opt_student.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_student.step()

#                 # Train discriminator (real vs student features)
#                 with torch.no_grad():
#                     s_feats_eval = self.FE(images)
#                 disc_real = self.discriminator(s_feats_eval).squeeze(1)
#                 disc_fake_labels = torch.zeros_like(disc_real)
#                 disc_loss = (self.bce_loss(disc_real, real_labels) + self.bce_loss(disc_real, disc_fake_labels)) / 2
#                 opt_disc.zero_grad()
#                 disc_loss.backward()
#                 opt_disc.step()

#         # Return updated FE + Classifier
#         return self.get_parameters(config), len(self.train_dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce_loss,
#             state='test',
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=10)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     # Setup
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     train_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_full = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)

#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(train_full, lab_idxs)
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgClient(args.cid, device, train_ds, None, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# fedavg plus al also for fedadagrad
# fl_client.py
# # fl_client.py
# import argparse
# import os, random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model

# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, unlab_ds, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_ds = train_ds
#         self.unlab_ds = unlab_ds
#         self.test_loader = test_loader
#         self.args = args

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)

#         lab_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds, batch_size=self.args.batch_size, shuffle=False)

#         opt_stu = optim.SGD(
#             list(self.FE.parameters()) +
#             list(self.classifier.parameters()) +
#             list(self.decoder.parameters()),
#             lr=self.args.lr_task,
#             momentum=0.9,
#             weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc,
#             momentum=0.9,
#             weight_decay=5e-4
#         )

#         max_grad_norm = 5.0

#         for _ in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             unlab_iter = iter(unlab_loader)

#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 opt_stu.zero_grad()
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 s_feats = self.FE(imgs)
#                 s_logits = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 loss_cls = self.ce(s_logits, labels)
#                 loss_feat = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 loss_adv = self.bce(disc_pred, torch.ones_like(disc_pred))

#                 loss_total = (
#                     loss_cls +
#                     self.args.distill_wt * (loss_feat + loss_logit) +
#                     self.args.adv_wt * loss_adv
#                 )

#                 loss_total.backward(retain_graph=True)
#                 torch.nn.utils.clip_grad_norm_(self.FE.parameters(), max_grad_norm)
#                 torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_grad_norm)
#                 torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_grad_norm)
#                 opt_stu.step()

#                 opt_disc.zero_grad()
#                 with torch.no_grad():
#                     feats_real = self.FE(imgs)

#                 try:
#                     unlb_imgs, _ = next(unlab_iter)
#                 except StopIteration:
#                     unlab_iter = iter(unlab_loader)
#                     try:
#                         unlb_imgs, _ = next(unlab_iter)
#                     except StopIteration:
#                         print(f"[Client {self.cid}] No unlabeled data left — skipping discriminator update")
#                         continue

#                 unlb_imgs = unlb_imgs.to(self.device)

#                 disc_real = self.discriminator(feats_real).squeeze(1)
#                 disc_fake = self.discriminator(self.FE(unlb_imgs)).squeeze(1)

#                 real_labels = torch.ones_like(disc_real)
#                 fake_labels = torch.zeros_like(disc_fake)

#                 sig_real = torch.sigmoid(disc_real).clamp(1e-7, 1 - 1e-7)
#                 sig_fake = torch.sigmoid(disc_fake).clamp(1e-7, 1 - 1e-7)

#                 loss_d = (self.bce(sig_real, real_labels) + self.bce(sig_fake, fake_labels)) / 2
#                 loss_d.backward()
#                 torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
#                 opt_disc.step()

#             self.active_learning_sampling(unlab_loader)

#         return self.get_parameters(config), len(self.train_ds), {}

#     def active_learning_sampling(self, unlab_loader):
#         self.FE.eval()
#         self.discriminator.eval()

#         all_feats = []
#         all_indices = []

#         with torch.no_grad():
#             for i, (imgs, _) in enumerate(unlab_loader):
#                 imgs = imgs.to(self.device)
#                 feats = self.FE(imgs)
#                 all_feats.append(feats.cpu())
#                 start_idx = i * self.args.batch_size
#                 end_idx = start_idx + len(imgs)
#                 all_indices.extend(range(start_idx, end_idx))

#         if all_feats:
#             all_feats = torch.cat(all_feats, 0)
#             scores = self.discriminator(all_feats.to(self.device)).detach().cpu().flatten()

#             k = min(self.args.budget, scores.shape[0])
#             _, sel = torch.topk(-scores, k)
#             selected_indices = [all_indices[i] for i in sel.numpy()]

#             self.train_ds.indices = np.concatenate([
#                 self.train_ds.indices,
#                 np.array(self.unlab_ds.indices)[selected_indices]
#             ])

#             mask = np.ones(len(self.unlab_ds.indices), dtype=bool)
#             mask[selected_indices] = False
#             self.unlab_ds.indices = np.array(self.unlab_ds.indices)[mask].tolist()

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test",
#         )
#         if np.isnan(loss):
#             loss = torch.tensor(1000.0)
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=20)
#     parser.add_argument("--lr_task", type=float, default=0.001)
#     parser.add_argument("--lr_disc", type=float, default=0.0001)
#     parser.add_argument("--distill_wt", type=float, default=0.5)
#     parser.add_argument("--adv_wt", type=float, default=0.01)
#     parser.add_argument("--budget", type=int, default=100)
#     parser.add_argument("--percent", type=float, default=0.1)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=100)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar100_transformer()
#     full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i = int(100 * args.percent)
#     lab = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))
#     train_ds = Subset(full, lab.tolist())
#     unlab_ds = Subset(full, unlab.tolist())

#     test_full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgALClient(args.cid, device, train_ds, unlab_ds, test_loader, args)

#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=client.to_client(),
#     )

# import argparse
# import os, random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model

# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, unlab_ds, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_ds = train_ds
#         self.unlab_ds = unlab_ds
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + AL modules
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         # Load global
#         self.set_parameters(parameters)

#         lab_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds, batch_size=self.args.batch_size, shuffle=False)

#         opt_stu = optim.SGD(
#             list(self.FE.parameters()) +
#             list(self.classifier.parameters()) +
#             list(self.decoder.parameters()),
#             lr=self.args.lr_task,
#             momentum=0.9,
#             weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc,
#             momentum=0.9,
#             weight_decay=5e-4
#         )
        
#         # Gradient clipping to prevent exploding gradients
#         max_grad_norm = 5.0

#         for _ in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             # Create iterator for unlabeled data
#             unlab_iter = iter(unlab_loader)
            
#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)
                
#                 # ===== Student Update =====
#                 opt_stu.zero_grad()
                
#                 # Teacher forward
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)
                
#                 # Student forward
#                 s_feats = self.FE(imgs)
#                 s_logits = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)
                
#                 # Align teacher spatial → student
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])
                
#                 # Losses
#                 loss_cls = self.ce(s_logits, labels)
#                 loss_feat = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
                
#                 # Adversarial loss
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 loss_adv = self.bce(disc_pred, torch.ones_like(disc_pred))
                
#                 # Total loss
#                 loss_total = (
#                     loss_cls +
#                     self.args.distill_wt * (loss_feat + loss_logit) +
#                     self.args.adv_wt * loss_adv
#                 )
                
#                 loss_total.backward(retain_graph=True)
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.FE.parameters(), max_grad_norm)
#                 torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_grad_norm)
#                 torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_grad_norm)
                
#                 opt_stu.step()

#                 # ===== Discriminator Update =====
#                 opt_disc.zero_grad()
                
#                 # Get features with new student
#                 with torch.no_grad():
#                     feats_real = self.FE(imgs)
                    
#                 # Get unlabeled batch
#                 try:
#                     unlb_imgs, _ = next(unlab_iter)
#                 except StopIteration:
#                     unlab_iter = iter(unlab_loader)
#                     unlb_imgs, _ = next(unlab_iter)
#                 unlb_imgs = unlb_imgs.to(self.device)
                
#                 # Discriminator outputs
#                 disc_real = self.discriminator(feats_real).squeeze(1)
#                 disc_fake = self.discriminator(self.FE(unlb_imgs)).squeeze(1)
                
#                 # Targets
#                 real_labels = torch.ones_like(disc_real)
#                 fake_labels = torch.zeros_like(disc_fake)
                
#                 # Apply sigmoid and clamp to prevent 0 and 1 exactly
#                 sig_real = torch.sigmoid(disc_real).clamp(1e-7, 1-1e-7)
#                 sig_fake = torch.sigmoid(disc_fake).clamp(1e-7, 1-1e-7)
                
#                 # BCE Loss with clamped values
#                 loss_d = (
#                     self.bce(sig_real, real_labels) +
#                     self.bce(sig_fake, fake_labels)
#                 ) / 2
                
#                 loss_d.backward()
                
#                 # Gradient clipping
#                 torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
                
#                 opt_disc.step()

#             # Active learning sampling
#             self.active_learning_sampling(unlab_loader)

#         return self.get_parameters(config), len(self.train_ds), {}
    
#     def active_learning_sampling(self, unlab_loader):
#         self.FE.eval()
#         self.discriminator.eval()
        
#         all_feats = []
#         all_indices = []
        
#         with torch.no_grad():
#             for i, (imgs, _) in enumerate(unlab_loader):
#                 imgs = imgs.to(self.device)
#                 feats = self.FE(imgs)
#                 all_feats.append(feats.cpu())
#                 # Track original indices
#                 start_idx = i * self.args.batch_size
#                 end_idx = start_idx + len(imgs)
#                 all_indices.extend(range(start_idx, end_idx))
        
#         if all_feats:
#             all_feats = torch.cat(all_feats, 0)
#             scores = self.discriminator(all_feats.to(self.device)).detach().cpu().flatten()
            
#             k = min(self.args.budget, scores.shape[0])
#             _, sel = torch.topk(-scores, k)
#             selected_indices = [all_indices[i] for i in sel.numpy()]
            
#             # Update datasets
#             self.train_ds.indices = np.concatenate([
#                 self.train_ds.indices, 
#                 np.array(self.unlab_ds.indices)[selected_indices]
#             ])
            
#             # Remove selected from unlabeled
#             mask = np.ones(len(self.unlab_ds.indices), dtype=bool)
#             mask[selected_indices] = False
#             self.unlab_ds.indices = np.array(self.unlab_ds.indices)[mask].tolist()

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test",
#         )
#         # Prevent NaN in evaluation
#         if np.isnan(loss):
#             loss = torch.tensor(1000.0)  # Large penalty for NaN
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",          type=int,   default=0)
#     parser.add_argument("--batch_size",   type=int,   default=64)
#     parser.add_argument("--local_epochs", type=int,   default=2)
#     parser.add_argument("--lr_task",      type=float, default=0.001)  # Reduced from 0.01
#     parser.add_argument("--lr_disc",      type=float, default=0.0001)  # Reduced from 0.001
#     parser.add_argument("--distill_wt",   type=float, default=0.5)  # Reduced from 1.0
#     parser.add_argument("--adv_wt",       type=float, default=0.01)  # Reduced from 0.1
#     parser.add_argument("--budget",       type=int,   default=100)  # Reduced from 500
#     parser.add_argument("--percent",      type=float, default=0.1)
#     parser.add_argument("--data_dir",     type=str,   default="data")
#     parser.add_argument("--num_classes",  type=int,   default=100)
#     parser.add_argument("--index_dir",    type=str,   default="client_indices")
#     args = parser.parse_args()

#     # reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # data + splits
#     transform = cifar100_transformer()
#     full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i = int(100 * args.percent)
#     lab   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))
#     train_ds = Subset(full, lab.tolist())
#     unlab_ds = Subset(full, unlab.tolist())

#     test_full   = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgALClient(args.cid, device, train_ds, unlab_ds, test_loader, args)
    
#     # Update to new Flower client initialization
#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=client.to_client(),
#     )
# # fl_client.py
# import argparse
# import os, random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar100_transformer, evaluate_model

# class FedAvgALClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, unlab_ds, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.train_ds = train_ds     # labeled Subset
#         self.unlab_ds = unlab_ds     # unlabeled Subset
#         self.test_loader = test_loader
#         self.args = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             r"C:\Users\nsola5\EnCoDe\Compressed Sampler\checkpoint_cifar100.pth",
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + AL modules
#         self.FE          = FeatureExtractor().to(device)
#         self.classifier  = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder     = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce  = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         # Load global
#         self.set_parameters(parameters)

#         lab_loader   = DataLoader(self.train_ds,      batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds,      batch_size=self.args.batch_size, shuffle=False)

#         opt_stu = optim.SGD(
#             list(self.FE.parameters()) +
#             list(self.classifier.parameters()) +
#             list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4
#         )

#         for _ in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train();         self.classifier.train()
#             self.decoder.train();    self.discriminator.train()

#             # student + adversarial updates
#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # teacher
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)
#                 # student
#                 s_feats   = self.FE(imgs)
#                 s_logits  = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)
#                 # align teacher spatial → student
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])
#                 # losses
#                 loss_cls   = self.ce(s_logits, labels)
#                 loss_feat  = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
#                 disc_pred  = self.discriminator(s_feats).squeeze(1)
#                 loss_adv   = self.bce(disc_pred, torch.ones_like(disc_pred, device=self.device))
#                 loss_total = (
#                     loss_cls
#                     + self.args.distill_wt * (loss_feat + loss_logit)
#                     + self.args.adv_wt * loss_adv
#                 )

#                 opt_stu.zero_grad()
#                 loss_total.backward(retain_graph=True)
#                 opt_stu.step()

#                 # discriminator step
#                 with torch.no_grad():
#                     feats_real = self.FE(imgs)

# # FAKE examples (random or zero noise as fake features)
#                 feats_fake = torch.randn_like(feats_real)

# # Discriminator outputs
#                 disc_real = self.discriminator(feats_real).squeeze(1)
#                 disc_fake = self.discriminator(feats_fake).squeeze(1)

# # Targets
#                 real_labels = torch.ones_like(disc_real)
#                 fake_labels = torch.zeros_like(disc_fake)

# # BCE Loss with sigmoid
#                 loss_d = (
#                     self.bce(disc_real, real_labels) +
#                     self.bce(disc_fake, fake_labels)
#                 ) / 2

#                 opt_disc.zero_grad()
#                 loss_d.backward()
#                 opt_disc.step()

#             # Active‐learning sampling
#             all_feats = []
#             for imgs, _ in unlab_loader:
#                 imgs = imgs.to(self.device)
#                 with torch.no_grad():
#                     all_feats.append(self.FE(imgs).cpu())
#             if all_feats:
#                 all_feats = torch.cat(all_feats, 0)
#                 scores   = self.discriminator(all_feats.to(self.device)).detach().cpu().flatten()
#                 k = min(self.args.budget, scores.shape[0])
#                 _, sel = torch.topk(-scores, k)
#                 new_idxs = np.array(self.unlab_ds.indices)[sel.numpy()]
#                 # add into labeled, remove from unlabeled
#                 self.train_ds.indices += new_idxs.tolist()
#                 mask = np.ones(len(self.unlab_ds.indices), dtype=bool)
#                 mask[sel] = False
#                 self.unlab_ds.indices = np.array(self.unlab_ds.indices)[mask].tolist()

#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",          type=int,   default=0)
#     parser.add_argument("--batch_size",   type=int,   default=64)
#     parser.add_argument("--local_epochs", type=int,   default=2)
#     parser.add_argument("--lr_task",      type=float, default=0.01)
#     parser.add_argument("--lr_disc",      type=float, default=0.001)
#     parser.add_argument("--distill_wt",   type=float, default=1.0)
#     parser.add_argument("--adv_wt",       type=float, default=0.1)
#     parser.add_argument("--budget",       type=int,   default=500)
#     parser.add_argument("--percent",      type=float, default=0.1)
#     parser.add_argument("--data_dir",     type=str,   default="data")
#     parser.add_argument("--num_classes",  type=int,   default=100)
#     parser.add_argument("--index_dir",    type=str,   default="client_indices")
#     args = parser.parse_args()

#     # reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # data + splits
#     transform = cifar100_transformer()
#     full = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i = int(100 * args.percent)
#     lab   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))
#     train_ds = Subset(full, lab.tolist())
#     unlab_ds = Subset(full, unlab.tolist())

#     test_full   = torchvision.datasets.CIFAR100(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAvgALClient(args.cid, device, train_ds, unlab_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# fl_client.py for fedprox without AL
# fl_client.py
# fl_client.py
# fl_client.py  (FedProx client, no AL)# fl_client.py
# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class FedProxClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, test_loader, args):
#         self.cid         = cid
#         self.device      = device
#         self.train_ds    = train_ds
#         self.test_loader = test_loader
#         self.args        = args

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + decoder + discriminator
#         self.FE           = FeatureExtractor().to(device)
#         self.classifier   = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder      = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce  = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         # Return FE + classifier parameters
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len    = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, new in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(new, device=self.device)
#         for p, new in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(new, device=self.device)

#     def fit(self, parameters, config):
#         # 1) Load global FE+classifier
#         self.set_parameters(parameters)
#         mu = config.get("mu", self.args.prox_mu)

#         # 2) Build labeled DataLoader
#         loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)

#         # 3) Optimizers
#         opt_stu = optim.SGD(
#             list(self.FE.parameters())
#             + list(self.classifier.parameters())
#             + list(self.decoder.parameters()),
#             lr=self.args.lr, momentum=0.9, weight_decay=5e-4,
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4,
#         )

#         # 4) Keep copy of global params for proximal term
#         global_params = [
#             torch.tensor(arr, device=self.device) for arr in parameters
#         ]

#         # 5) Local training
#         self.teacher.eval()
#         for _ in range(self.args.local_epochs):
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             for imgs, labels in loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # Teacher forward
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 # Student forward
#                 s_feats   = self.FE(imgs)
#                 s_logits  = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)

#                 # Pool teacher→student spatial dims
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 # Losses
#                 loss_cls   = self.ce(s_logits, labels)
#                 loss_feat  = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
#                 disc_pred  = self.discriminator(s_feats).squeeze(1)
#                 loss_adv   = self.bce(disc_pred, torch.ones_like(disc_pred))

#                 # Proximal regularization
#                 prox = 0.0
#                 for p, g in zip(
#                     list(self.FE.parameters()) + list(self.classifier.parameters()),
#                     global_params,
#                 ):
#                     prox += torch.sum((p - g) ** 2)

#                 loss = (
#                     loss_cls
#                     + self.args.distill_wt * (loss_feat + loss_logit)
#                     + self.args.adv_wt * loss_adv
#                     + 0.5 * mu * prox
#                 )

#                 # Update student + decoder
#                 opt_stu.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_stu.step()

#                 # Update discriminator
#                 with torch.no_grad():
#                     feats_eval = self.FE(imgs)
#                 pred = self.discriminator(feats_eval).squeeze(1)
#                 real, fake = torch.ones_like(pred), torch.zeros_like(pred)
#                 loss_d = (self.bce(pred, real) + self.bce(pred, fake)) / 2
#                 opt_disc.zero_grad()
#                 loss_d.backward()
#                 opt_disc.step()

#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",        type=int,   default=0)
#     parser.add_argument("--batch_size", type=int,   default=64)
#     parser.add_argument("--local_epochs", type=int, default=5)
#     parser.add_argument("--lr",         type=float, default=0.01)
#     parser.add_argument("--lr_disc",    type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt",     type=float, default=0.1)
#     parser.add_argument("--prox_mu",    type=float, default=0.1)
#     parser.add_argument("--percent",    type=float, default=1.0)
#     parser.add_argument("--data_dir",   type=str,   default="data")
#     parser.add_argument("--num_classes",type=int,   default=10)
#     parser.add_argument("--index_dir",  type=str,   default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     full = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i = int(100 * args.percent)
#     lab   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     train_ds = Subset(full, lab.tolist())

#     test_full   = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedProxClient(args.cid, device, train_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# fl_client.py  (FedProx client + Active Learning)
# fl_client.py
# fl_client.py
# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# #
# # ── SubsetWithIndices ─────────────────────────────────────────────────────────
# #
# class SubsetWithIndices(Dataset):
#     """A dataset wrapper which returns (x, y, idx) for each sample."""
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         real_idx = self.indices[i]
#         x, y = self.dataset[real_idx]
#         return x, y, real_idx


# #
# # ── FedProxClient (with AL) ────────────────────────────────────────────────────
# #
# class FedProxClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, full_dataset, lab_idxs, unlab_idxs, test_loader, args):
#         self.cid         = cid
#         self.device      = device
#         self.args        = args

#         # Build labeled & unlabeled subsets
#         self.train_ds    = torch.utils.data.Subset(full_dataset, lab_idxs)
#         self.unlab_ds    = SubsetWithIndices(full_dataset, unlab_idxs)
#         self.test_loader = test_loader

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + decoder + discriminator
#         self.FE            = FeatureExtractor().to(device)
#         self.classifier    = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder       = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce  = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len    = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, arr in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)

#     def fit(self, parameters, config):
#         # 1) Load global weights
#         self.set_parameters(parameters)
#         mu = config.get("mu", self.args.prox_mu)

#         # 2) DataLoaders
#         lab_loader   = DataLoader(self.train_ds,  batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds,  batch_size=self.args.batch_size, shuffle=False)

#         # 3) Optimizers
#         opt_stu = optim.SGD(
#             list(self.FE.parameters())
#             + list(self.classifier.parameters())
#             + list(self.decoder.parameters()),
#             lr=self.args.lr, momentum=0.9, weight_decay=5e-4,
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4,
#         )

#         # 4) Snapshot global params for FedProx
#         global_params = [torch.tensor(arr, device=self.device) for arr in parameters]

#         # 5) Local training
#         self.teacher.eval()
#         for _ in range(self.args.local_epochs):
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # Teacher forward
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 # Student + decoder forward
#                 s_feats   = self.FE(imgs)
#                 s_logits  = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)

#                 # Align spatial dims
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 # Losses
#                 loss_cls   = self.ce(s_logits, labels)
#                 loss_feat  = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
#                 disc_pred  = self.discriminator(s_feats).squeeze(1)
#                 loss_adv   = self.bce(disc_pred, torch.ones_like(disc_pred))

#                 # Proximal regularization
#                 prox = 0.0
#                 for p, g in zip(
#                     list(self.FE.parameters()) + list(self.classifier.parameters()),
#                     global_params,
#                 ):
#                     prox += torch.sum((p - g) ** 2)

#                 loss = (
#                     loss_cls
#                     + self.args.distill_wt * (loss_feat + loss_logit)
#                     + self.args.adv_wt   * loss_adv
#                     + 0.5 * mu            * prox
#                 )

#                 # Backprop student + decoder
#                 opt_stu.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_stu.step()

#                 # Discriminator update
#                 with torch.no_grad():
#                     feats_eval = self.FE(imgs)
#                 pred = self.discriminator(feats_eval).squeeze(1)
#                 real, fake = torch.ones_like(pred), torch.zeros_like(pred)
#                 loss_d = (self.bce(pred, real) + self.bce(pred, fake)) / 2
#                 opt_disc.zero_grad()
#                 loss_d.backward()
#                 opt_disc.step()

#             # ── Active Learning sampling ──
#             self.FE.eval()
#             self.discriminator.eval()
#             all_scores, all_idxs = [], []
#             with torch.no_grad():
#                 for imgs, _, idxs in unlab_loader:
#                     feats = self.FE(imgs.to(self.device))
#                     scores = self.discriminator(feats).squeeze(1).cpu()
#                     all_scores.append(scores)
#                     all_idxs.append(idxs)
#             if all_scores:
#                 all_scores = torch.cat(all_scores)
#                 all_idxs   = torch.cat(all_idxs)
#                 k = min(self.args.budget, len(all_scores))
#                 _, sel = torch.topk(-all_scores, k)
#                 picked = all_idxs[sel].tolist()
#                 # move into labeled set
#                 self.train_ds.indices += picked
#                 # remove from unlabeled
#                 self.unlab_ds.indices = [i for i in self.unlab_ds.indices if i not in picked]

#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE        = self.FE,
#             classifier= self.classifier,
#             loader    = self.test_loader,
#             device    = self.device,
#             criterion = self.ce,
#             state     = "test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",          type=int,   default=0)
#     parser.add_argument("--batch_size",   type=int,   default=64)
#     parser.add_argument("--local_epochs", type=int,   default=5)
#     parser.add_argument("--lr",           type=float, default=0.01)
#     parser.add_argument("--lr_disc",      type=float, default=0.001)
#     parser.add_argument("--distill_wt",   type=float, default=1.0)
#     parser.add_argument("--adv_wt",       type=float, default=0.1)
#     parser.add_argument("--prox_mu",      type=float, default=0.1)
#     parser.add_argument("--budget",       type=int,   default=500)
#     parser.add_argument("--percent",      type=float, default=1.0)
#     parser.add_argument("--data_dir",     type=str,   default="data")
#     parser.add_argument("--num_classes",  type=int,   default=10)
#     parser.add_argument("--index_dir",    type=str,   default="client_indices")
#     args = parser.parse_args()

#     # Reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Data + splits
#     transform = cifar10_transformer()
#     full      = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i     = int(100 * args.percent)
#     lab_idxs   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))

#     test_full   = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedProxClient(
#         cid        = args.cid,
#         device     = device,
#         full_dataset= full,
#         lab_idxs    = lab_idxs.tolist(),
#         unlab_idxs  = unlab_idxs.tolist(),
#         test_loader = test_loader,
#         args        = args,
#     )
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision
# import flwr as fl
# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class SubsetWithIndices(Dataset):
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         real_idx = self.indices[i]
#         x, y = self.dataset[real_idx]
#         return x, y, real_idx

# class FedProxClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, full_dataset, lab_idxs, unlab_idxs, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.args = args

#         self.train_ds = torch.utils.data.Subset(full_dataset, lab_idxs)
#         self.unlab_ds = SubsetWithIndices(full_dataset, unlab_idxs)
#         self.test_loader = test_loader

#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         self.ce = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#         self.informative_round = False  # Track if this client was informative in this round

#     def get_parameters(self, config):
#         if not self.informative_round:
#             return []
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         if len(parameters) == 0:
#             return
#         fe_len = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, arr in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         mu = config.get("mu", self.args.prox_mu)

#         lab_loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds, batch_size=self.args.batch_size, shuffle=False)

#         opt_stu = optim.SGD(
#             list(self.FE.parameters()) + list(self.classifier.parameters()) + list(self.decoder.parameters()),
#             lr=self.args.lr, momentum=0.9, weight_decay=5e-4,
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4,
#         )

#         global_params = [torch.tensor(arr, device=self.device) for arr in parameters]

#         self.teacher.eval()
#         for _ in range(self.args.local_epochs):
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 s_feats = self.FE(imgs)
#                 s_logits = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)

#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 loss_cls = self.ce(s_logits, labels)
#                 loss_feat = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)
#                 disc_pred = self.discriminator(s_feats).squeeze(1)
#                 loss_adv = self.bce(disc_pred, torch.ones_like(disc_pred))

#                 prox = 0.0
#                 for p, g in zip(list(self.FE.parameters()) + list(self.classifier.parameters()), global_params):
#                     prox += torch.sum((p - g) ** 2)

#                 loss = (
#                     loss_cls
#                     + self.args.distill_wt * (loss_feat + loss_logit)
#                     + self.args.adv_wt * loss_adv
#                     + 0.5 * mu * prox
#                 )

#                 opt_stu.zero_grad()
#                 loss.backward(retain_graph=True)
#                 opt_stu.step()

#                 with torch.no_grad():
#                     feats_eval = self.FE(imgs)
#                 pred = self.discriminator(feats_eval).squeeze(1)
#                 real, fake = torch.ones_like(pred), torch.zeros_like(pred)
#                 loss_d = (self.bce(pred, real) + self.bce(pred, fake)) / 2
#                 opt_disc.zero_grad()
#                 loss_d.backward()
#                 opt_disc.step()

#         self.FE.eval()
#         self.discriminator.eval()
#         all_scores, all_idxs = [], []
#         with torch.no_grad():
#             for imgs, _, idxs in unlab_loader:
#                 feats = self.FE(imgs.to(self.device))
#                 scores = self.discriminator(feats).squeeze(1).cpu()
#                 all_scores.append(scores)
#                 all_idxs.append(idxs)

#         self.informative_round = False
#         if all_scores:
#             all_scores = torch.cat(all_scores)
#             all_idxs = torch.cat(all_idxs)
#             k = min(self.args.budget, len(all_scores))
#             _, sel = torch.topk(-all_scores, k)
#             picked = all_idxs[sel].tolist()
#             if picked:
#                 self.informative_round = True
#             self.train_ds.indices += picked
#             self.unlab_ds.indices = [i for i in self.unlab_ds.indices if i not in picked]

#         return self.get_parameters(config), len(self.train_ds) if self.informative_round else 0, {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=5)
#     parser.add_argument("--lr", type=float, default=0.01)
#     parser.add_argument("--lr_disc", type=float, default=0.001)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--prox_mu", type=float, default=0.1)
#     parser.add_argument("--budget", type=int, default=500)
#     parser.add_argument("--percent", type=float, default=1.0)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     full = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))

#     test_full = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedProxClient(
#         cid=args.cid,
#         device=device,
#         full_dataset=full,
#         lab_idxs=lab_idxs.tolist(),
#         unlab_idxs=unlab_idxs.tolist(),
#         test_loader=test_loader,
#         args=args,
#     )
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


# FEDPROX with al
# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class SubsetWithIndices(Dataset):
#     """Dataset that returns (x, y, idx) so we can re-sample unlabeled examples."""
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         real_idx = self.indices[i]
#         x, y = self.dataset[real_idx]
#         return x, y, real_idx


# class FedAdamClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, full_dataset, lab_idxs, unlab_idxs, test_loader, args):
#         self.cid            = cid
#         self.device         = device
#         self.args           = args

#         # Labeled / unlabeled splits
#         self.train_ds       = Subset(full_dataset, lab_idxs)
#         self.unlab_ds       = SubsetWithIndices(full_dataset, unlab_idxs)
#         self.test_loader    = test_loader

#         # Frozen teacher
#         self.teacher        = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student + decoder + discriminator
#         self.FE             = FeatureExtractor().to(device)
#         self.classifier     = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder        = Decoder().to(device)
#         self.discriminator  = Discriminator().to(device)

#         # Losses
#         self.ce_loss  = nn.CrossEntropyLoss()
#         self.bce_loss = nn.BCELoss()
#         self.mse_loss = nn.MSELoss()

#     def get_parameters(self, config):
#         # Only FE + classifier go to the server
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len    = len(list(self.FE.parameters()))
#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:]
#         for p, arr in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)

#     def fit(self, parameters, config):
#         # 1) Load server parameters
#         self.set_parameters(parameters)

#         # 2) DataLoaders
#         lab_loader   = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)
#         unlab_loader = DataLoader(self.unlab_ds, batch_size=self.args.batch_size, shuffle=False)

#         # 3) Opts
#         opt_student = optim.SGD(
#             list(self.FE.parameters())
#             + list(self.classifier.parameters())
#             + list(self.decoder.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4,
#         )
#         opt_disc = optim.SGD(
#             self.discriminator.parameters(),
#             lr=self.args.lr_disc, momentum=0.9, weight_decay=5e-4,
#         )

#         # 4) Local epochs
#         for _ in range(self.args.local_epochs):
#             self.teacher.eval()
#             self.FE.train(); self.classifier.train()
#             self.decoder.train(); self.discriminator.train()

#             # a) supervised + distillation + adversarial
#             for imgs, labels in lab_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # teacher
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 # student
#                 s_feats     = self.FE(imgs)
#                 s_logits    = self.classifier(s_feats)
#                 dec_feats   = self.decoder(s_feats)
#                 t_pooled    = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 # losses
#                 loss_cls    = self.ce_loss(s_logits, labels)
#                 loss_feat   = self.mse_loss(dec_feats, t_pooled)
#                 loss_logit  = self.mse_loss(s_logits, t_logits)
#                 disc_pred   = self.discriminator(s_feats).squeeze(1)
#                 loss_adv    = self.bce_loss(disc_pred, torch.ones_like(disc_pred))

#                 total_loss = (
#                     loss_cls
#                     + self.args.distill_wt * (loss_feat + loss_logit)
#                     + self.args.adv_wt   * loss_adv
#                 )

#                 opt_student.zero_grad()
#                 total_loss.backward(retain_graph=True)
#                 opt_student.step()

#                 # discriminator step
#                 with torch.no_grad():
#                     feats_eval = self.FE(imgs)
#                 pred = self.discriminator(feats_eval).squeeze(1)
#                 real, fake = torch.ones_like(pred), torch.zeros_like(pred)
#                 loss_d = (self.bce_loss(pred, real) + self.bce_loss(pred, fake)) / 2
#                 opt_disc.zero_grad()
#                 loss_d.backward()
#                 opt_disc.step()

#             # b) active-learning sampling
#             self.FE.eval(); self.discriminator.eval()
#             all_scores, all_idxs = [], []
#             with torch.no_grad():
#                 for imgs, _, idxs in unlab_loader:
#                     feats  = self.FE(imgs.to(self.device))
#                     scores = self.discriminator(feats).squeeze(1).cpu()
#                     all_scores.append(scores)
#                     all_idxs.append(idxs)
#             if all_scores:
#                 all_scores = torch.cat(all_scores)
#                 all_idxs   = torch.cat(all_idxs)
#                 k          = min(self.args.budget, len(all_scores))
#                 # pick the k *most* uncertain (lowest disc confidence)
#                 _, sel     = torch.topk(-all_scores, k)
#                 picked     = all_idxs[sel].tolist()
#                 # add to labeled
#                 self.train_ds.indices += picked
#                 # remove from unlabeled
#                 self.unlab_ds.indices = [i for i in self.unlab_ds.indices if i not in picked]

#         # 5) Return new FE+classifier
#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE         = self.FE,
#             classifier = self.classifier,
#             loader     = self.test_loader,
#             device     = self.device,
#             criterion  = self.ce_loss,
#             state      = "test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",          type=int,   default=0)
#     parser.add_argument("--batch_size",   type=int,   default=64)
#     parser.add_argument("--local_epochs", type=int,   default=1)
#     parser.add_argument("--lr_task",      type=float, default=0.01)
#     parser.add_argument("--lr_disc",      type=float, default=0.001)
#     parser.add_argument("--distill_wt",   type=float, default=1.0)
#     parser.add_argument("--adv_wt",       type=float, default=0.1)
#     parser.add_argument("--budget",       type=int,   default=100)
#     parser.add_argument("--percent",      type=float, default=1.0)
#     parser.add_argument("--data_dir",     type=str,   default="data")
#     parser.add_argument("--num_classes",  type=int,   default=10)
#     parser.add_argument("--index_dir",    type=str,   default="client_indices")
#     args = parser.parse_args()

#     # reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # data + splits
#     transform  = cifar10_transformer()
#     full_train = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i      = int(100 * args.percent)
#     lab_idxs   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct_i}.npy"))
#     unlab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_split_{pct_i}.npy"))

#     test_full  = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAdamClient(
#         cid            = args.cid,
#         device         = device,
#         full_dataset   = full_train,
#         lab_idxs       = lab_idxs.tolist(),
#         unlab_idxs     = unlab_idxs.tolist(),
#         test_loader    = test_loader,
#         args           = args,
#     )
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
# fl_client.py for fedadam plus al 
# import argparse
# import os
# import random

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model


# class SubsetWithIndices(Dataset):
#     """Dataset that returns (x, y, idx) so we can re-sample unlabeled examples."""
#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = list(indices)

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         real_idx = self.indices[i]
#         x, y = self.dataset[real_idx]
#         return x, y, real_idx


# class FedAdamClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, full_dataset, lab_idxs, unlab_idxs, test_loader, args):
#         self.cid         = cid
#         self.device      = device
#         self.args        = args
#         # labeled / unlabeled
#         self.train_ds    = Subset(full_dataset, lab_idxs)
#         self.unlab_ds    = SubsetWithIndices(full_dataset, unlab_idxs)
#         self.test_loader = test_loader

#         # frozen teacher
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(
#             os.path.join("checkpoints", "checkpoint.pth"),
#             map_location="cpu",
#             weights_only=True,
#         )
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # student + decoder + discriminator
#         self.fe            = FeatureExtractor().to(device)
#         self.cl            = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder       = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # losses
#         self.ce  = nn.CrossEntropyLoss()
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def get_parameters(self, config):
#         # only FE + classifier
#         params = [p.detach().cpu().numpy() for p in self.fe.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.cl.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.discriminator.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.fe.parameters()))
#         cl_len = len(list(self.cl.parameters()))
#         disc_len = len(list(self.discriminator.parameters()))

#         fe_params   = parameters[:fe_len]
#         cl_params   = parameters[fe_len:fe_len + cl_len]
#         disc_params = parameters[fe_len + cl_len:]

#         for p, arr in zip(self.fe.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.cl.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.discriminator.parameters(), disc_params):
#             p.data = torch.tensor(arr, device=self.device)


#     def fit(self, parameters, config):
#         # 1) load global FE+Classifier
#         self.set_parameters(parameters)

#         # 2) build loaders
#         sup_loader   = DataLoader(self.train_ds,   batch_size=self.args.batch_size, shuffle=True)
#         unsup_loader = DataLoader(self.unlab_ds,   batch_size=self.args.batch_size, shuffle=False)

#         # 3) one optimizer for all modules
#         opt = optim.SGD(
#             list(self.fe.parameters())
#           + list(self.cl.parameters())
#           + list(self.decoder.parameters())
#           + list(self.discriminator.parameters()),
#             lr=self.args.lr_task,
#             momentum=0.9,
#             weight_decay=5e-4,
#         )

#         self.teacher.eval()
#         for _ in range(self.args.local_epochs):
#             self.fe.train()
#             self.cl.train()
#             self.decoder.train()
#             self.discriminator.train()

#             # a) supervised + distill + adv + disc self-sup
#             for imgs, labels in sup_loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # teacher outputs
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 # student forward
#                 s_feats   = self.fe(imgs)
#                 s_logits  = self.cl(s_feats)
#                 dec_feats = self.decoder(s_feats)
#                 t_pooled  = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 # losses
#                 loss_sup   = self.ce(s_logits, labels)
#                 loss_dist  = self.mse(dec_feats, t_pooled) + self.mse(s_logits, t_logits)
#                 pred_adv   = self.discriminator(s_feats).squeeze(1)
#                 pred_adv = torch.sigmoid(pred_adv)
#                 loss_adv   = self.bce(pred_adv, torch.ones_like(pred_adv))

#                 # discriminator self-supervision on the same batch
#                 with torch.no_grad():
#                     feats_eval = self.fe(imgs)
#                 d_eval = self.discriminator(feats_eval).squeeze(1)
#                 d_eval = torch.sigmoid(d_eval)
#                 loss_d = 0.5 * (
#                     self.bce(d_eval, torch.ones_like(d_eval))
#                   + self.bce(d_eval, torch.zeros_like(d_eval))
#                 )

#                 # total
#                 loss = (
#                     loss_sup
#                   + self.args.distill_wt * loss_dist
#                   + self.args.adv_wt     * loss_adv
#                   + self.args.d_wt       * loss_d
#                 )

#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()

#             # b) active-learning sampling
#             self.fe.eval()
#             self.discriminator.eval()
#             all_scores, all_idxs = [], []
#             with torch.no_grad():
#                 for imgs, _, idxs in unsup_loader:
#                     feats  = self.fe(imgs.to(self.device))
#                     scores = self.discriminator(feats).squeeze(1).cpu()
#                     all_scores.append(scores)
#                     all_idxs.append(idxs)
#             if all_scores:
#                 all_scores = torch.cat(all_scores)
#                 all_idxs   = torch.cat(all_idxs)
#                 k          = min(self.args.budget, len(all_scores))
#                 _, sel     = torch.topk(-all_scores, k)
#                 picked     = all_idxs[sel].tolist()
#                 # add to labeled set
#                 self.train_ds.indices += picked
#                 # remove from unlabeled
#                 self.unlab_ds.indices = [i for i in self.unlab_ds.indices if i not in picked]

#         # return updated FE+Classifier
#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE         = self.fe,
#             classifier = self.cl,
#             loader     = self.test_loader,
#             device     = self.device,
#             criterion  = self.ce,
#             state      = "test",
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid",         type=int,   default=0)
#     parser.add_argument("--batch_size",  type=int,   default=64)
#     parser.add_argument("--local_epochs",type=int,   default=10)
#     parser.add_argument("--lr_task",     type=float, default=0.01)
#     parser.add_argument("--distill_wt",  type=float, default=1.0)
#     parser.add_argument("--adv_wt",      type=float, default=0.1)
#     parser.add_argument("--d_wt",        type=float, default=0.1,
#                         help="weight for discriminator self-supervision")
#     parser.add_argument("--lr_disc",     type=float, default=0.001)  # still unused—kept for backward compatibility
#     parser.add_argument("--budget",      type=int,   default=100)
#     parser.add_argument("--percent",     type=float, default=1.0)
#     parser.add_argument("--data_dir",    type=str,   default="data")
#     parser.add_argument("--num_classes", type=int,   default=10)
#     parser.add_argument("--index_dir",   type=str,   default="client_indices")
#     args = parser.parse_args()

#     # reproducibility
#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform  = cifar10_transformer()
#     full_train = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     pct_i      = int(100 * args.percent)
#     lab_idxs   = np.load(os.path.join(args.index_dir, f"client_{args.cid}_labeled_{pct_i}.npy"))
#     unlab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_unlabeled_{pct_i}.npy"))

#     test_full  = torchvision.datasets.CIFAR10(
#         root=args.data_dir, train=False, download=True, transform=transform
#     )
#     test_loader = DataLoader(test_full, batch_size=args.batch_size, shuffle=False)

#     client = FedAdamClient(
#         cid          = args.cid,
#         device       = device,
#         full_dataset = full_train,
#         lab_idxs     = lab_idxs.tolist(),
#         unlab_idxs   = unlab_idxs.tolist(),
#         test_loader  = test_loader,
#         args         = args,
#     )
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# # fedadam without al
# import argparse
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import flwr as fl

# from models import (
#     VGG,
#     CompVGGFeature as FeatureExtractor,
#     CompVGGClassifier as Classifier,
#     Decoder,
#     Discriminator,
# )
# from utils import cifar10_transformer, evaluate_model

# class FedAdamClient(fl.client.NumPyClient):
#     def __init__(self, cid, device, train_ds, test_loader, args):
#         self.cid = cid
#         self.device = device
#         self.args = args
#         self.train_ds = train_ds
#         self.test_loader = test_loader

#         # Teacher (frozen)
#         self.teacher = VGG(num_classes=args.num_classes).to(device)
#         ckpt = torch.load(os.path.join("checkpoints", "checkpoint.pth"), map_location="cpu", weights_only=True)
#         self.teacher.load_state_dict(ckpt, strict=True)
#         self.teacher.eval()

#         # Student components
#         self.FE = FeatureExtractor().to(device)
#         self.classifier = Classifier(num_classes=args.num_classes).to(device)
#         self.decoder = Decoder().to(device)
#         self.discriminator = Discriminator().to(device)

#         # Losses
#         self.ce = nn.CrossEntropyLoss()
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCELoss()

#     def get_parameters(self, config):
#         params = [p.detach().cpu().numpy() for p in self.FE.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.classifier.parameters()]
#         params += [p.detach().cpu().numpy() for p in self.discriminator.parameters()]
#         return params

#     def set_parameters(self, parameters):
#         fe_len = len(list(self.FE.parameters()))
#         cl_len = len(list(self.classifier.parameters()))
#         disc_len = len(list(self.discriminator.parameters()))

#         fe_params = parameters[:fe_len]
#         cl_params = parameters[fe_len:fe_len + cl_len]
#         disc_params = parameters[fe_len + cl_len:]

#         for p, arr in zip(self.FE.parameters(), fe_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.classifier.parameters(), cl_params):
#             p.data = torch.tensor(arr, device=self.device)
#         for p, arr in zip(self.discriminator.parameters(), disc_params):
#             p.data = torch.tensor(arr, device=self.device)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         loader = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True)

#         # Optimizer
#         opt = optim.SGD(
#             list(self.FE.parameters()) +
#             list(self.classifier.parameters()) +
#             list(self.decoder.parameters()) +
#             list(self.discriminator.parameters()),
#             lr=self.args.lr_task, momentum=0.9, weight_decay=5e-4
#         )

#         self.teacher.eval()
#         for _ in range(self.args.local_epochs):
#             self.FE.train()
#             self.classifier.train()
#             self.decoder.train()
#             self.discriminator.train()

#             for imgs, labels in loader:
#                 imgs, labels = imgs.to(self.device), labels.to(self.device)

#                 # Teacher
#                 with torch.no_grad():
#                     t_feats, t_logits = self.teacher(imgs)

#                 # Student forward
#                 s_feats = self.FE(imgs)
#                 s_logits = self.classifier(s_feats)
#                 dec_feats = self.decoder(s_feats)
#                 t_pooled = F.adaptive_avg_pool2d(t_feats, dec_feats.shape[2:])

#                 # Losses
#                 loss_cls = self.ce(s_logits, labels)
#                 loss_feat = self.mse(dec_feats, t_pooled)
#                 loss_logit = self.mse(s_logits, t_logits)

#                 pred_adv = self.discriminator(s_feats).squeeze(1)
#                 pred_adv = torch.sigmoid(pred_adv)
#                 loss_adv = self.bce(pred_adv, torch.ones_like(pred_adv))

#                 # Self-supervised discriminator loss
#                 with torch.no_grad():
#                     feats_eval = self.FE(imgs)
#                 disc_pred = self.discriminator(feats_eval).squeeze(1)
#                 disc_pred = torch.sigmoid(disc_pred)
#                 loss_d = 0.5 * (self.bce(disc_pred, torch.ones_like(disc_pred)) +
#                                 self.bce(disc_pred, torch.zeros_like(disc_pred)))

#                 # Final loss
#                 total_loss = (
#                     loss_cls +
#                     self.args.distill_wt * (loss_feat + loss_logit) +
#                     self.args.adv_wt * loss_adv +
#                     self.args.d_wt * loss_d
#                 )

#                 opt.zero_grad()
#                 total_loss.backward()
#                 opt.step()

#         return self.get_parameters(config), len(self.train_ds), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate_model(
#             FE=self.FE,
#             classifier=self.classifier,
#             loader=self.test_loader,
#             device=self.device,
#             criterion=self.ce,
#             state="test"
#         )
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cid", type=int, default=0)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--local_epochs", type=int, default=10)
#     parser.add_argument("--lr_task", type=float, default=0.01)
#     parser.add_argument("--distill_wt", type=float, default=1.0)
#     parser.add_argument("--adv_wt", type=float, default=0.1)
#     parser.add_argument("--d_wt", type=float, default=0.1)
#     parser.add_argument("--data_dir", type=str, default="data")
#     parser.add_argument("--num_classes", type=int, default=10)
#     parser.add_argument("--index_dir", type=str, default="client_indices")
#     parser.add_argument("--percent", type=float, default=1.0)
#     args = parser.parse_args()

#     random.seed(args.cid)
#     np.random.seed(args.cid)
#     torch.manual_seed(args.cid)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = cifar10_transformer()
#     full_train = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
#     test_set = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

#     pct = int(100 * args.percent)
#     lab_idxs = np.load(os.path.join(args.index_dir, f"client_{args.cid}_split_{pct}.npy"))
#     train_ds = Subset(full_train, lab_idxs.tolist())

#     client = FedAdamClient(args.cid, device, train_ds, test_loader, args)
#     fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
