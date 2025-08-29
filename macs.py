# from ptflops import get_model_complexity_info
# import torch
# import torch.nn as nn
# from models import CompVGGFeature as FeatureExtractor, CompVGGClassifier as Classifier

# # Step 1: Instantiate and check output shape of Feature Extractor
# fe = FeatureExtractor()
# sample_input = torch.randn(1, 3, 32, 32)
# with torch.no_grad():
#     feature_output = fe(sample_input)
# print("✅ Feature extractor output shape:", feature_output.shape)

# # Extract the number of channels/features
# feature_dim = feature_output.shape[1]

# # Step 2: Classifier Wrapper with the correct expected input size
# class ClassifierWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.classifier = Classifier(num_classes=10)

#     def forward(self, x):
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         if x.dim() == 2:
#             x = x.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
#         return self.classifier(x)

# # Step 3: Get MACs for Classifier
# try:
#     macs_cls, params_cls = get_model_complexity_info(
#         ClassifierWrapper(),
#         (feature_dim, 1, 1),  # Correct input shape from FeatureExtractor
#         as_strings=False,
#         print_per_layer_stat=False,
#     )
# except Exception as e:
#     print("⚠️ Could not compute MACs for classifier:", e)
#     macs_cls, params_cls = 0, 0

# # Step 4: Get MACs for Feature Extractor
# macs_fe, params_fe = get_model_complexity_info(
#     fe,
#     (3, 32, 32),
#     as_strings=False,
#     print_per_layer_stat=False,
# )

# # Step 5: Display Results
# total_macs = macs_fe + macs_cls
# print(f"\n📊 MACs Report:")
# print(f"Feature Extractor MACs: {macs_fe:,}")
# print(f"Classifier MACs:        {macs_cls:,}")
# print(f"Total MACs per sample:  {total_macs:,}")
# import torch
# import torch.nn as nn
# from ptflops import get_model_complexity_info
# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Instantiate models
# feature_extractor = CompVGGFeature().to(device)
# classifier = CompVGGClassifier(num_classes=10).to(device)
# decoder = Decoder().to(device)
# discriminator = Discriminator().to(device)

# # Safe MACs computation helper
# def get_macs_safe(model, input_shape):
#     try:
#         macs, params = get_model_complexity_info(
#             model,
#             input_shape,
#             as_strings=False,
#             print_per_layer_stat=False,
#         )
#         return macs, params
#     except Exception as e:
#         print(f"❌ Error with model {model.__class__.__name__}: {e}")
#         return 0, 0

# # --- 1. Feature Extractor ---
# macs_fe, params_fe = get_macs_safe(feature_extractor, (3, 32, 32))

# # --- 2. Classifier ---
# macs_cls, params_cls = get_macs_safe(classifier, (64, 2, 2))

# # --- 3. Decoder ---
# class DecoderWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder = Decoder()

#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#         return self.decoder(x)

# macs_dec, params_dec = get_macs_safe(DecoderWrapper(), (64, 2, 2))

# # --- 4. Discriminator (expects feature input of size [1, 256]) ---
# class DiscriminatorWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.discriminator = Discriminator()

#     def forward(self, x):
#         return self.discriminator(x)

# macs_disc, params_disc = get_macs_safe(DiscriminatorWrapper(), (256,))

# # --- Total MACs per sample ---
# total_sampler_macs = macs_fe + macs_cls + macs_dec + macs_disc

# # --- Reporting ---
# print(f"\n📊 MACs Summary:")
# print(f"Feature Extractor MACs: {macs_fe:,}")
# print(f"Classifier MACs:        {macs_cls:,}")
# print(f"Decoder MACs:           {macs_dec:,}")
# print(f"Discriminator MACs:     {macs_disc:,}")
# print(f"✅ Total Active Learning MACs per sample: {total_sampler_macs:,}")

# # --- Total Training MACs (1000 samples × 5 epochs) ---
# samples = 1000
# epochs = 5
# total_training_ops = total_sampler_macs * samples * epochs
# print(f"\n📈 Total Training MACs (1000 samples × 5 epochs): {total_training_ops:,}")

# # --- Communication Cost (FeatureExtractor + Classifier only) ---
# params_to_server = sum(p.numel() for p in list(feature_extractor.parameters()) + list(classifier.parameters()))
# print(f"\n📤 Params sent to server:     {params_to_server:,}")
# print(f"📥 Params received from server: {params_to_server:,}")
# import torch
# import torch.nn as nn
# from ptflops import get_model_complexity_info
# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Instantiate models
# feature_extractor = CompVGGFeature().to(device)
# classifier = CompVGGClassifier(num_classes=10).to(device)
# decoder = Decoder().to(device)
# discriminator = Discriminator().to(device)

# # Safe MACs computation helper
# def get_macs_safe(model, input_shape):
#     try:
#         macs, params = get_model_complexity_info(
#             model,
#             input_shape,
#             as_strings=False,
#             print_per_layer_stat=False,
#         )
#         return macs, params
#     except Exception as e:
#         print(f"❌ Error with model {model.__class__.__name__}: {e}")
#         return 0, 0

# # --- MACs per sample ---
# macs_fe, params_fe     = get_macs_safe(feature_extractor, (3, 32, 32))
# macs_cls, params_cls   = get_macs_safe(classifier, (64, 2, 2))

# class DecoderWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder = Decoder()
#     def forward(self, x):
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#         return self.decoder(x)

# macs_dec, params_dec = get_macs_safe(DecoderWrapper(), (64, 2, 2))

# class DiscriminatorWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.discriminator = Discriminator()
#     def forward(self, x):
#         return self.discriminator(x)

# macs_disc, params_disc = get_macs_safe(DiscriminatorWrapper(), (256,))  # discriminator input = flattened features

# # --- Categorize MACs ---
# sampler_macs_per_sample  = macs_fe + macs_dec + macs_disc         # used during uncertainty sampling
# training_macs_per_sample = macs_fe + macs_cls + macs_dec + macs_disc  # full training loop as per fl_client

# # --- Estimate training cost (1000 samples × 5 epochs × 2 for fwd+back) ---
# samples = 1
# epochs  = 10
# training_total_ops = training_macs_per_sample * samples * epochs * 2

# # --- Communication cost: only FE + Classifier
# params_to_server = sum(p.numel() for p in list(feature_extractor.parameters()) + list(classifier.parameters()))

# # --- Report ---
# print(f"\n📊 MACs Summary (per sample):")
# print(f"Feature Extractor:  {macs_fe:,}")
# print(f"Classifier:         {macs_cls:,}")
# print(f"Decoder:            {macs_dec:,}")
# print(f"Discriminator:      {macs_disc:,}")

# print(f"\n🔍 Sampler MACs per sample: {sampler_macs_per_sample:,}")
# print(f"🧠 Training MACs per sample (fwd only): {training_macs_per_sample:,}")
# print(f"📈 Total Training MACs (1000 samples × 5 epochs × fwd+back): {training_total_ops:,}")

# print(f"\n📤 Params sent to server:     {params_to_server:,}")
# print(f"📥 Params received from server: {params_to_server:,}")
# import torch
# import torch.nn as nn
# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Instantiate models
# feature_extractor = CompVGGFeature().to(device)
# classifier = CompVGGClassifier(num_classes=10).to(device)
# decoder = Decoder().to(device)
# discriminator = Discriminator().to(device)

# # Dummy inputs (match training setup)
# x_img = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)
# x_feat = torch.randn(1, 64, 2, 2, requires_grad=True).to(device)
# x_flat = torch.randn(1, 256, requires_grad=True).to(device)

# loss_fn = nn.MSELoss()

# def measure_real_macs(model, input_tensor, target_tensor, name):
#     input_tensor = input_tensor.clone().detach().requires_grad_(True)
#     model = model.to(device)
#     output = model(input_tensor)
#     loss = loss_fn(output, target_tensor)
#     loss.backward()

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     input_elements = input_tensor.numel()
#     output_elements = output.numel()
    
#     # Approximate: 2x params + IO ops (forward + backward)
#     total_macs = 2 * total_params + input_elements + output_elements

#     print(f"{name}:\n"
#           f"  Trainable Params: {total_params:,}\n"
#           f"  Input Elements:   {input_elements:,}\n"
#           f"  Output Elements:  {output_elements:,}\n"
#           f"  Total Fwd+Bwd MACs: {total_macs:,}\n")
#     return total_macs

# print("📊 Measuring Actual Forward + Backward MACs per Sample\n")

# # Feature Extractor
# target_feat = torch.zeros((1, 64, 2, 2)).to(device)
# macs_fe = measure_real_macs(feature_extractor, x_img, target_feat, "Feature Extractor")

# # Classifier
# target_cls = torch.zeros((1, 10)).to(device)
# macs_cl = measure_real_macs(classifier, x_feat, target_cls, "Classifier")

# # Decoder (target shape inferred from decoder output)
# with torch.no_grad():
#     test_out = decoder(x_feat)
# target_dec = torch.zeros_like(test_out)
# macs_dec = measure_real_macs(decoder, x_feat, target_dec, "Decoder")

# # Discriminator
# target_disc = torch.zeros((1, 1)).to(device)
# macs_disc = measure_real_macs(discriminator, x_flat, target_disc, "Discriminator")

# # Total MACs
# total_macs = macs_fe + macs_cl + macs_dec + macs_disc
# print(f"✅ Total Real MACs for One Training Pass (1 sample): {total_macs:,}")
# import torch
# import torch.nn as nn
# from torch.profiler import profile, ProfilerActivity, record_function
# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Instantiate models
# FE = CompVGGFeature().to(device)
# CL = CompVGGClassifier(num_classes=10).to(device)
# DEC = Decoder().to(device)
# DISC = Discriminator().to(device)

# # Dummy inputs (match your setup)
# x_img  = torch.randn(1, 3, 32, 32).to(device)
# x_feat = torch.randn(1, 64, 2, 2).to(device)
# x_flat = torch.randn(1, 256).to(device)

# mse = nn.MSELoss()

# def profile_block(name, model, x, target):
#     model.train()
#     x = x.clone().detach().requires_grad_(True)
    
#     with profile(
#         activities=[ProfilerActivity.CPU],  # No CUDA since it's unavailable
#         with_flops=True,
#         profile_memory=False
#     ) as prof:
#         with record_function(name):
#             out = model(x)
#             loss = mse(out, target)
#             loss.backward()
    
#     total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
#     print(f"{name} FLOPs (fwd + bwd): {total_flops:,}")
#     return total_flops


# # 1️⃣ Sampler: FE + DEC + DISC
# print("\n🔍 1. Running Sampler MACs:")
# with torch.no_grad(): target_dec = DEC(x_feat)
# target_disc = torch.zeros((1, 1)).to(device)

# sampler_flops = 0
# sampler_flops += profile_block("FeatureExtractor", FE, x_img, torch.zeros((1, 64, 2, 2)).to(device))
# sampler_flops += profile_block("Decoder", DEC, x_feat, target_dec)
# sampler_flops += profile_block("Discriminator", DISC, x_flat, target_disc)

# # 2️⃣ Training Model: FE + CL + DEC + DISC
# print("\n🧠 2. Training Model MACs:")
# target_cls = torch.zeros((1, 10)).to(device)

# training_flops = 0
# training_flops += profile_block("FeatureExtractor", FE, x_img, torch.zeros((1, 64, 2, 2)).to(device))
# training_flops += profile_block("Classifier", CL, x_feat, target_cls)
# training_flops += profile_block("Decoder", DEC, x_feat, target_dec)
# training_flops += profile_block("Discriminator", DISC, x_flat, target_disc)

# # 3️⃣ Params sent to server
# params_to_server = sum(p.numel() for p in list(FE.parameters()) + list(CL.parameters()))
# print(f"\n📤 3. Params sent to server: {params_to_server:,}")

# # 4️⃣ Params received from server (same)
# print(f"📥 4. Params received from server: {params_to_server:,}")


# import torch
# import torch.nn as nn
# from torch.profiler import profile, ProfilerActivity, record_function
# from models import CompVGGFeature, CompVGGClassifier, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === Initialize models ===
# FE_sampler = CompVGGFeature().to(device)
# DISC       = Discriminator().to(device)

# FE_train = CompVGGFeature().to(device)
# CL_train = CompVGGClassifier(num_classes=10).to(device)

# mse = nn.MSELoss()

# # === Dummy Inputs ===
# x_img   = torch.randn(1, 3, 32, 32).to(device)
# x_feat  = FE_sampler(x_img).detach()
# x_flat  = torch.flatten(x_feat, 1).detach()
# target  = torch.zeros_like(x_flat).to(device)

# # === FLOPs profiling helper ===
# def profile_block_split_fwd_bwd(name, model, x, target):
#     model.train()
#     x = x.clone().detach().requires_grad_(True)

#     with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
#         with record_function(f"{name}_forward"):
#             out = model(x)
#             loss = mse(out, target)
#         with record_function(f"{name}_backward"):
#             loss.backward()

#     fwd_flops = sum(evt.flops for evt in prof.key_averages() if f"{name}_forward" in evt.key and evt.flops)
#     bwd_flops = sum(evt.flops for evt in prof.key_averages() if f"{name}_backward" in evt.key and evt.flops)

#     return fwd_flops, bwd_flops

# # === Sampler FLOPs ===
# print("\n🔍 Sampler (FeatureExtractor + Discriminator):")
# sampler_fwd, sampler_bwd = 0, 0
# f, b = profile_block_split_fwd_bwd("Sampler_FE", FE_sampler, x_img, x_feat); sampler_fwd += f; sampler_bwd += b
# f, b = profile_block_split_fwd_bwd("Sampler_DISC", DISC, x_flat, torch.zeros((1, 1)).to(device)); sampler_fwd += f; sampler_bwd += b

# print(f"🔸 Sampler Forward FLOPs:  {sampler_fwd:,}")
# print(f"🔸 Sampler Backward FLOPs: {sampler_bwd:,}")
# print(f"🔸 Sampler Total FLOPs:    {sampler_fwd + sampler_bwd:,}")

# # === Training FLOPs ===
# print("\n🧠 Node Training (FeatureExtractor + Classifier):")
# train_fwd, train_bwd = 0, 0
# x_feat_train = FE_train(x_img).detach()
# target_cls = torch.zeros((1, 10)).to(device)

# f, b = profile_block_split_fwd_bwd("Train_FE", FE_train, x_img, x_feat_train); train_fwd += f; train_bwd += b
# f, b = profile_block_split_fwd_bwd("Train_CL", CL_train, x_feat_train, target_cls); train_fwd += f; train_bwd += b

# print(f"🔸 Training Forward FLOPs:  {train_fwd:,}")
# print(f"🔸 Training Backward FLOPs: {train_bwd:,}")
# print(f"🔸 Training Total FLOPs:    {train_fwd + train_bwd:,}")

# # === Communication Cost ===
# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# params_sent = count_params(FE_train) + count_params(CL_train) + count_params(DISC)
# params_recv = params_sent

# print(f"\n📡 Communication Cost:")
# print(f"📤 Parameters sent to server:     {params_sent:,}")
# print(f"📥 Parameters received from server: {params_recv:,}")

# # Breakdown
# print(f"\n🔍 Breakdown:")
# print(f"  - Feature Extractor: {count_params(FE_train):,}")
# print(f"  - Classifier:        {count_params(CL_train):,}")
# print(f"  - Discriminator:     {count_params(DISC):,}")
# import torch
# import torch.nn as nn
# from torch.profiler import profile, ProfilerActivity, record_function
# import pandas as pd
# from models import CompVGGFeature, CompVGGClassifier, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === Initialize models ===
# FE_sampler = CompVGGFeature().to(device)
# DISC       = Discriminator().to(device)
# FE_train   = CompVGGFeature().to(device)
# CL_train   = CompVGGClassifier(num_classes=10).to(device)

# mse = nn.MSELoss()

# # === Dummy Inputs ===
# x_img   = torch.randn(1, 3, 32, 32).to(device)
# x_feat  = FE_sampler(x_img).detach()
# x_flat  = torch.flatten(x_feat, 1).detach()
# target  = torch.zeros_like(x_flat).to(device)  # fake target
# target_cls = torch.zeros((1, 10)).to(device)

# # === FLOPs profiling helper ===
# def profile_block_split_fwd_bwd(name, model, x, target):
#     model.train()
#     x = x.clone().detach().requires_grad_(True)
    
#     with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
#         with record_function(f"{name}_forward"):
#             out = model(x)
#             loss = mse(out, target)
#         with record_function(f"{name}_backward"):
#             loss.backward()

#     fwd_flops = sum(evt.flops for evt in prof.key_averages() if f"{name}_forward" in evt.key and evt.flops)
#     bwd_flops = sum(evt.flops for evt in prof.key_averages() if f"{name}_backward" in evt.key and evt.flops)
#     return fwd_flops, bwd_flops

# # === 🔍 Sampler FLOPs ===
# sampler_fwd, sampler_bwd = 0, 0
# f, b = profile_block_split_fwd_bwd("Sampler_FE", FE_sampler, x_img, x_feat); sampler_fwd += f; sampler_bwd += b
# f, b = profile_block_split_fwd_bwd("Sampler_DISC", DISC, x_flat, torch.zeros((1, 1)).to(device)); sampler_fwd += f; sampler_bwd += b

# # === 🧠 Node Training FLOPs ===
# train_fwd, train_bwd = 0, 0
# x_feat_train = FE_train(x_img).detach()
# f, b = profile_block_split_fwd_bwd("Train_FE", FE_train, x_img, x_feat_train); train_fwd += f; train_bwd += b
# f, b = profile_block_split_fwd_bwd("Train_CL", CL_train, x_feat_train, target_cls); train_fwd += f; train_bwd += b

# # === 📡 Communication Costs ===
# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# params_sent = count_params(FE_train) + count_params(CL_train) + count_params(DISC)
# params_recv = params_sent  # same modules returned

# # === 📊 Table Display ===
# data = {
#     "Module": ["Sampler (FE + DISC)", "Training (FE + CL)", "Discriminator"],
#     "Forward FLOPs": [sampler_fwd, train_fwd, 0],
#     "Backward FLOPs": [sampler_bwd, train_bwd, 0],
#     "Total FLOPs": [sampler_fwd + sampler_bwd, train_fwd + train_bwd, 0],
#     "Params": [count_params(FE_sampler) + count_params(DISC), count_params(FE_train) + count_params(CL_train), count_params(DISC)],
# }

# df = pd.DataFrame(data)
# print("\n=== FLOPs and Parameter Breakdown ===")
# print(df.to_string(index=False))

# print("\n📡 Communication Cost Summary:")
# print(f"📤 Parameters sent to server:     {params_sent:,}")
# print(f"📥 Parameters received from server: {params_recv:,}")
# import torch
# from ptflops import get_model_complexity_info
# from models import CompVGGFeature, CompVGGClassifier, Discriminator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dummy input shape (C, H, W) for CIFAR-10
# input_shape = (3, 32, 32)

# # Initialize models
# fe_model = CompVGGFeature().to(device)
# cl_model = CompVGGClassifier(num_classes=10).to(device)
# disc_model = Discriminator().to(device)

# print("=== FLOPs and Parameter Breakdown ===")

# def print_flops_params(name, model, input_shape):
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
#     flops = 2 * macs  # MACs * 2 = FLOPs
#     print(f"{name:<20} FLOPs: {flops:,} | Params: {params:,}")
#     return flops, params

# # Sampler = FeatureExtractor + Discriminator
# sampler_flops, sampler_params = 0, 0
# f, p = print_flops_params("Feature Extractor", fe_model, input_shape); sampler_flops += f; sampler_params += p
# f, p = print_flops_params("Discriminator", disc_model, (64, 2, 2)); sampler_flops += f; sampler_params += p

# # Trainer = FeatureExtractor + Classifier
# trainer_flops, trainer_params = 0, 0
# f, p = print_flops_params("Feature Extractor", fe_model, input_shape); trainer_flops += f; trainer_params += p
# f, p = print_flops_params("Classifier", cl_model, (64, 2, 2)); trainer_flops += f; trainer_params += p

# print(f"\n📊 Sampler Total FLOPs: {sampler_flops:,} | Params: {sampler_params:,}")
# print(f"📊 Trainer Total FLOPs: {trainer_flops:,} | Params: {trainer_params:,}")
# print(f"📦 Discriminator Only   : {f:,} | Params: {p:,}")

# # Communication cost (FE + CL + DISC)
# comm_params = count = sum(p.numel() for p in fe_model.parameters()) + \
#                       sum(p.numel() for p in cl_model.parameters()) + \
#                       sum(p.numel() for p in disc_model.parameters())
# print(f"\n📡 Communication (FE+CL+DISC): {comm_params:,} parameters sent/received")
# import torch
# from ptflops import get_model_complexity_info
# from models import CompVGGFeature, CompVGGClassifier, Discriminator, Decoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dummy input shape (C, H, W) for CIFAR-10
# input_shape = (3, 32, 32)

# # Initialize models
# fe_model = CompVGGFeature().to(device)
# cl_model = CompVGGClassifier(num_classes=10).to(device)
# disc_model = Discriminator().to(device)
# dec_model = Decoder().to(device)

# print("=== FLOPs and Parameter Breakdown ===")

# def print_flops_params(name, model, input_shape):
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
#     flops = 2 * macs  # MACs * 2 = FLOPs
#     print(f"{name:<20} FLOPs: {flops:,} | Params: {params:,}")
#     return flops, params

# # Sampler = FeatureExtractor + Discriminator
# sampler_flops, sampler_params = 0, 0
# f, p = print_flops_params("Feature Extractor", fe_model, input_shape); sampler_flops += f; sampler_params += p
# f, p = print_flops_params("Discriminator", disc_model, (64, 2, 2)); sampler_flops += f; sampler_params += p

# # Trainer = FeatureExtractor + Classifier + Decoder
# trainer_flops, trainer_params = 0, 0
# f, p = print_flops_params("Feature Extractor", fe_model, input_shape); trainer_flops += f; trainer_params += p
# f, p = print_flops_params("Classifier", cl_model, (64, 2, 2)); trainer_flops += f; trainer_params += p
# f, p = print_flops_params("Decoder", dec_model, (64, 2, 2)); trainer_flops += f; trainer_params += p

# print(f"\n📊 Sampler Total FLOPs: {sampler_flops:,} | Params: {sampler_params:,}")
# print(f"📊 Trainer Total FLOPs: {trainer_flops:,} | Params: {trainer_params:,}")
# print(f"📦 Decoder Only         : {f:,} | Params: {p:,}")

# # Communication cost (FE + CL + DISC)
# comm_params = (
#     sum(p.numel() for p in fe_model.parameters()) +
#     sum(p.numel() for p in cl_model.parameters()) +
#     sum(p.numel() for p in disc_model.parameters())
# # )
# # print(f"\n📡 Communication (FE+CL+DISC): {comm_params:,} parameters sent/received")
# import torch
# from ptflops import get_model_complexity_info
# from models import CompVGGFeature, CompVGGClassifier, Discriminator, Decoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dummy input shape (C, H, W) for CIFAR-10
# input_shape = (3, 32, 32)

# # Initialize models
# fe_model   = CompVGGFeature().to(device)
# cl_model   = CompVGGClassifier(num_classes=10).to(device)
# disc_model = Discriminator().to(device)
# dec_model  = Decoder().to(device)

# print("=== FLOPs and Parameter Breakdown ===")

# def print_flops_params(name, model, input_shape):
#     macs, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
#     flops = 2 * macs  # MACs * 2 = FLOPs
#     print(f"{name:<20} FLOPs: {flops:,} | Params: {params:,}")
#     return flops, params

# # FLOPs for Sampler = FeatureExtractor + Discriminator
# sampler_flops, sampler_params = 0, 0
# f, p = print_flops_params("Sampler FE", fe_model, input_shape); sampler_flops += f; sampler_params += p
# f, p = print_flops_params("Discriminator", disc_model, (64, 2, 2)); sampler_flops += f; sampler_params += p

# # FLOPs for Trainer = FeatureExtractor + Classifier + Decoder
# trainer_flops, trainer_params = 0, 0
# f, p = print_flops_params("Trainer FE", fe_model, input_shape); trainer_flops += f; trainer_params += p
# f, p = print_flops_params("Classifier", cl_model, (64, 2, 2)); trainer_flops += f; trainer_params += p
# f, p = print_flops_params("Decoder", dec_model, (64, 2, 2)); trainer_flops += f; trainer_params += p

# print(f"\n📊 Sampler Total FLOPs      : {sampler_flops:,} | Params: {sampler_params:,}")
# print(f"📊 Trainer Total FLOPs      : {trainer_flops:,} | Params: {trainer_params:,}")
# print(f"📦 Decoder Only FLOPs       : {f:,} | Params: {p:,}")

# # Communication cost (params exchanged = FE + CL + DISC)
# comm_params = (
#     sum(p.numel() for p in fe_model.parameters()) +
#     sum(p.numel() for p in cl_model.parameters()) +
#     sum(p.numel() for p in disc_model.parameters())
# )
# print(f"\n📡 Communication (FE+CL+DISC): {comm_params:,} parameters sent/received per round")
# import matplotlib.pyplot as plt
# import numpy as np

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Values
# comp_without_al = [314.08e9, 244.29e9, 279.18e9, 209.39e9]  # in MACs
# comp_with_al    = [174.91e9, 139.92e9, 104.94e9, 139.92e9]
# comm_without_al = [4324.44] * 4  # in KB
# comm_with_al    = [4453.33] * 4

# # Plot setup
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Actual Bars
# bar1 = ax1.bar(x - bar_width / 2, comm_without_al, bar_width, color='#6baed6')
# bar2 = ax1.bar(x + bar_width / 2, comp_without_al, bar_width, color='#fcae91')
# bar3 = ax2.bar(x - bar_width / 2, comm_with_al, bar_width, color='#2171b5', alpha=0.5)
# bar4 = ax2.bar(x + bar_width / 2, comp_with_al, bar_width, color='#cb181d', alpha=0.5)

# # Axis labels
# ax1.set_ylabel('Communication Cost (KB)')
# ax2.set_ylabel('Computation Cost (MACs)')
# ax1.set_title('Dual Y-Axis Cost Comparison (With Legend)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques)

# # Explicit legend handles
# custom_lines = [
#     plt.Rectangle((0, 0), 1, 1, color='#6baed6'),  # Comm (w/o AL)
#     plt.Rectangle((0, 0), 1, 1, color='#fcae91'),  # Comm (with AL)
#     plt.Rectangle((0, 0), 1, 1, color='#2171b5', alpha=0.5),  # Comp (w/o AL)
#     plt.Rectangle((0, 0), 1, 1, color='#cb181d', alpha=0.5)   # Comp (with AL)
# ]
# labels = ['Comm (w/o AL)', 'Comm (with AL)', 'Comp (w/o AL)', 'Comp (with AL)']
# ax1.legend(custom_lines, labels, loc='upper left', bbox_to_anchor=(1.01, 1))

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.25

# # Values
# comm_without_al = [4324.44] * 4  # in KB
# comm_with_al    = [4453.33] * 4
# comp_without_al = [314.08e9, 244.29e9, 279.18e9, 209.39e9]  # in MACs
# comp_with_al    = [174.91e9, 139.92e9, 104.94e9, 139.92e9]

# # Plot setup
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Positions for bars
# comm_pos = x - bar_width / 2
# comp_pos = x + bar_width / 2

# # Plot stacked bars
# bar1 = ax1.bar(comm_pos, comm_without_al, bar_width, label='Comm (w/o AL)', color="#9ecae1")
# bar2 = ax1.bar(comm_pos, comm_with_al, bar_width, bottom=comm_without_al, label='Comm (with AL)', color="#08519c")

# bar3 = ax2.bar(comp_pos, comp_without_al, bar_width, label='Comp (w/o AL)', color="#fbb4b9")
# bar4 = ax2.bar(comp_pos, comp_with_al, bar_width, bottom=comp_without_al, label='Comp (with AL)', color="#a50f15")

# # Axis labels
# ax1.set_ylabel('Communication Cost (KB)', color='#08519c')
# ax2.set_ylabel('Computation Cost (MACs)', color='#a50f15')
# ax1.set_title('Communication and Computation Cost per Technique (Stacked Bars)')
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques)

# # Legend
# lines = [bar1[0], bar2[0], bar3[0], bar4[0]]
# labels = ['Comm (w/o AL)', 'Comm (with AL)', 'Comp (w/o AL)', 'Comp (with AL)']
# ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.01, 1))

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))  # no spacing multiplier

# # Values
# comm_without_al = [4324.44] * 4  # KB
# comm_with_al    = [4453.33] * 4
# comp_without_al = [314.08e9, 244.29e9, 279.18e9, 209.39e9]  # MACs
# comp_with_al    = [174.91e9, 139.92e9, 104.94e9, 139.92e9]

# bar_width = 0.35

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Communication Bars (left bar in group)
# ax1.bar(x - bar_width / 2, comm_without_al, width=bar_width, color='lightgrey', hatch='//', label='Comm (w/o AL)')
# ax1.bar(x - bar_width / 2, comm_with_al, width=bar_width, color='dimgrey', hatch='\\\\', bottom=comm_without_al, label='Comm (with AL)')

# # Computation Bars (right bar in group)
# ax2.bar(x + bar_width / 2, comp_without_al, width=bar_width, color='lightgrey', hatch='..', label='Comp (w/o AL)')
# ax2.bar(x + bar_width / 2, comp_with_al, width=bar_width, color='dimgrey', hatch='xx', bottom=comp_without_al, label='Comp (with AL)')

# # Axes labels and title
# ax1.set_ylabel('Communication Cost (KB)', fontsize=12)
# ax2.set_ylabel('Computation Cost (MACs)', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques, fontsize=11)
# ax1.set_title('Communication and Computation Cost per Technique', fontsize=14)

# # Legend
# legend_elements = [
#     Patch(facecolor='lightgrey', hatch='//', label='Comm (w/o AL)'),
#     Patch(facecolor='dimgrey', hatch='\\\\', label='Comm (with AL)'),
#     Patch(facecolor='lightgrey', hatch='..', label='Comp (w/o AL)'),
#     Patch(facecolor='dimgrey', hatch='xx', label='Comp (with AL)')
# ]
# ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))  # index positions

# # Values
# comm_without_al = [4324.44] * 4  # Communication cost without AL (KB)
# comm_with_al = [3237.57, 3651.73, 3504.00, 3584.995]  # Updated values with AL (KB)

# comp_without_al = [314.08e9, 244.29e9, 279.18e9, 209.39e9]  # Computation cost without AL (MACs)
# comp_with_al =    [174.91e9, 139.92e9, 104.94e9, 139.92e9]  # Computation cost with AL (MACs)

# bar_width = 0.35

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Communication Bars
# ax1.bar(x - bar_width / 2, comm_without_al, width=bar_width, color='lightgrey', hatch='//', label='Comm (w/o AL)')
# ax1.bar(x - bar_width / 2, comm_with_al, width=bar_width, color='dimgrey', hatch='\\\\', label='Comm (with AL)')

# # Computation Bars
# ax2.bar(x + bar_width / 2, comp_without_al, width=bar_width, color='lightgrey', hatch='..', label='Comp (w/o AL)')
# ax2.bar(x + bar_width / 2, comp_with_al, width=bar_width, color='dimgrey', hatch='xx', label='Comp (with AL)')

# # Labels and Title
# ax1.set_ylabel('Communication Cost (Normalized)', fontsize=12)
# ax2.set_ylabel('Computation Cost (Normalized)', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques, fontsize=11)
# ax1.set_title('Communication and Computation Cost per Technique', fontsize=14)

# # Legend
# legend_elements = [
#     Patch(facecolor='lightgrey', hatch='//', label='Comm (w/o AL)'),
#     Patch(facecolor='dimgrey', hatch='\\\\', label='Comm (with AL)'),
#     Patch(facecolor='lightgrey', hatch='..', label='Comp (w/o AL)'),
#     Patch(facecolor='dimgrey', hatch='xx', label='Comp (with AL)')
# ]
# ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Energy values (in mJ)
# comm_energy_wo_al = [172.98] * 4
# comm_energy_w_al = [129.50, 146.07, 140.16, 143.40]
# comp_energy_wo_al = [31408, 24429, 27918, 20939]
# comp_energy_w_al = [17491, 13992, 10494, 13992]

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # Bars for AL
# ax1.bar(x - bar_width/2, comm_energy_w_al, width=bar_width, color='gray', hatch='//', edgecolor='black')
# ax2.bar(x - bar_width/2, comp_energy_w_al, width=bar_width, color='black', hatch='xx', edgecolor='black', bottom=comm_energy_w_al)

# # Bars for w/o AL
# ax1.bar(x + bar_width/2, comm_energy_wo_al, width=bar_width, color='lightgray', hatch='\\\\', edgecolor='black')
# ax2.bar(x + bar_width/2, comp_energy_wo_al, width=bar_width, color='dimgray', hatch='..', edgecolor='black', bottom=comm_energy_wo_al)

# # Labels and ticks
# ax1.set_ylabel('Communication Energy (mJ)', fontsize=12)
# ax2.set_ylabel('Computation Energy (mJ)', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques, fontsize=11)
# ax1.set_title('Communication and Computation Energy per Technique (AL vs w/o AL)', fontsize=14)

# # Legend patches
# legend_elements = [
#     Patch(facecolor='lightgray', hatch='\\\\', edgecolor='black', label='Comm (w/o AL)'),
#     Patch(facecolor='gray', hatch='//', edgecolor='black', label='Comm (with AL)'),
#     Patch(facecolor='dimgray', hatch='..', edgecolor='black', label='Comp (w/o AL)'),
#     Patch(facecolor='black', hatch='xx', edgecolor='black', label='Comp (with AL)'),
# ]

# # Place legend neatly above the plot
# ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# # Adjust layout to not clip legend
# plt.tight_layout()
# plt.show()
# Re-import libraries after code state reset
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Normalized Energy Values
# comp_with_al = [0.556896332, 0.445491594, 0.334118696, 0.445491594]
# comp_without_al = [1, 0.777795466, 0.888881814, 0.66667728]
# comm_with_al = [0.00105555, 0.001190579, 0.001142415, 0.001168822]
# comm_without_al = [0.001409904] * 4

# fig, ax = plt.subplots(figsize=(10, 6))

# # Bars for With AL
# ax.bar(x - bar_width/2, comm_with_al, width=bar_width, color='gray', hatch='//', edgecolor='black')
# ax.bar(x - bar_width/2, comp_with_al, width=bar_width, color='black', hatch='xx', edgecolor='black', bottom=comm_with_al)

# # Bars for Without AL
# ax.bar(x + bar_width/2, comm_without_al, width=bar_width, color='lightgray', hatch='\\\\', edgecolor='black')
# ax.bar(x + bar_width/2, comp_without_al, width=bar_width, color='dimgray', hatch='..', edgecolor='black', bottom=comm_without_al)

# # Labels and ticks
# ax.set_ylabel('Normalized Energy Cost (Computation + Communication)', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(techniques, fontsize=11)
# ax.set_title('Normalized Computation and Communication Cost per Technique (AL vs w/o AL)', fontsize=14)

# # Legend patches
# legend_elements = [
#     Patch(facecolor='lightgray', hatch='\\\\', edgecolor='black', label='Comm (w/o AL)'),
#     Patch(facecolor='gray', hatch='//', edgecolor='black', label='Comm (with AL)'),
#     Patch(facecolor='dimgray', hatch='..', edgecolor='black', label='Comp (w/o AL)'),
#     Patch(facecolor='black', hatch='xx', edgecolor='black', label='Comp (with AL)'),
# ]

# # Add legend
# ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Updated actual energy data (raw values)
# # Computation: MACs, Communication: Bytes
# comp_energy_w_al = [1.7491e12, 1.3992e12, 1.0494e12, 1.3992e12]
# comp_energy_wo_al = [3.1408e12, 2.4429e12, 2.7918e12, 2.0939e12]
# comm_energy_w_al = [3315271680, 3739371520, 3588096000, 3671034880]
# comm_energy_wo_al = [4428226560] * 4

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # With AL
# ax1.bar(x - bar_width / 2, comm_energy_w_al, width=bar_width, color='gray', hatch='//', edgecolor='black')
# ax2.bar(x - bar_width / 2, comp_energy_w_al, width=bar_width, color='black', hatch='xx', edgecolor='black', bottom=comm_energy_w_al)

# # Without AL
# ax1.bar(x + bar_width / 2, comm_energy_wo_al, width=bar_width, color='lightgray', hatch='\\\\', edgecolor='black')
# ax2.bar(x + bar_width / 2, comp_energy_wo_al, width=bar_width, color='dimgray', hatch='..', edgecolor='black', bottom=comm_energy_wo_al)

# # Labels and ticks
# ax1.set_ylabel('Communication (Bytes)', fontsize=12)
# ax2.set_ylabel('Computation (MACs)', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques, fontsize=11)
# ax1.set_title('Communication and Computation per Technique (AL vs w/o AL)', fontsize=14)

# # Legend
# legend_elements = [
#     Patch(facecolor='lightgray', hatch='\\\\', edgecolor='black', label='Comm (w/o AL)'),
#     Patch(facecolor='gray', hatch='//', edgecolor='black', label='Comm (with AL)'),
#     Patch(facecolor='dimgray', hatch='..', edgecolor='black', label='Comp (w/o AL)'),
#     Patch(facecolor='black', hatch='xx', edgecolor='black', label='Comp (with AL)'),
# ]
# ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Patch

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Raw energy values
# # Computation in MACs, Communication in Bytes
# comp_energy_w_al = [1.7491e12, 1.3992e12, 1.0494e12, 1.3992e12]
# comp_energy_wo_al = [3.1408e12, 2.4429e12, 2.7918e12, 2.0939e12]
# comm_energy_w_al = [3315271680, 3739371520, 3588096000, 3671034880]
# comm_energy_wo_al = [4428226560] * 4

# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax2 = ax1.twinx()

# # ==== Stack Communication OVER Computation ====

# # With AL
# ax2.bar(x - bar_width / 2, comp_energy_w_al, width=bar_width, color='black', hatch='xx', edgecolor='black')
# ax1.bar(x - bar_width / 2, comm_energy_w_al, width=bar_width, color='gray', hatch='//', edgecolor='black', bottom=comp_energy_w_al)

# # Without AL
# ax2.bar(x + bar_width / 2, comp_energy_wo_al, width=bar_width, color='dimgray', hatch='..', edgecolor='black')
# ax1.bar(x + bar_width / 2, comm_energy_wo_al, width=bar_width, color='lightgray', hatch='\\\\', edgecolor='black', bottom=comp_energy_wo_al)

# # Labels and ticks
# ax1.set_ylabel('Communication (Bytes)', fontsize=12)
# ax2.set_ylabel('Computation (MACs)', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques, fontsize=11)
# ax1.set_title('Communication and Computation per Technique (AL vs w/o AL)', fontsize=14)

# # Legend
# legend_elements = [
#     Patch(facecolor='lightgray', hatch='\\\\', edgecolor='black', label='Comm (w/o AL)'),
#     Patch(facecolor='gray', hatch='//', edgecolor='black', label='Comm (with AL)'),
#     Patch(facecolor='dimgray', hatch='..', edgecolor='black', label='Comp (w/o AL)'),
#     Patch(facecolor='black', hatch='xx', edgecolor='black', label='Comp (with AL)'),
# ]
# ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Techniques
techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
x = np.arange(len(techniques))
bar_width = 0.35

# Data from the image (normalized)
comp_with_al     = [0.556896332, 0.445491594, 0.334118696, 0.445491594]
comp_without_al  = [1.0,         0.777795466, 0.888881814, 0.66667728]
comm_with_al     = [0.0105555,  0.01190579, 0.01142415, 0.01168822]
comm_without_al  = [0.001409904] * 4

fig, ax = plt.subplots(figsize=(10, 6))

# WITH AL: computation bottom, comm stacked on top
ax.bar(x - bar_width/2, comp_with_al, width=bar_width, color='black', hatch='xx', edgecolor='black')
ax.bar(x - bar_width/2, comm_with_al, width=bar_width, bottom=comp_with_al, color='gray', hatch='//', edgecolor='black')

# WITHOUT AL: computation bottom, comm stacked on top
ax.bar(x + bar_width/2, comp_without_al, width=bar_width, color='dimgray', hatch='..', edgecolor='black')
ax.bar(x + bar_width/2, comm_without_al, width=bar_width, bottom=comp_without_al, color='lightgray', hatch='\\\\', edgecolor='black')

# Labels and axis
ax.set_ylabel('Normalized Total Energy (Computation + Communication)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(techniques, fontsize=11)
ax.set_title('Stacked Computation + Communication per Technique (FLARE vs BASELINE )', fontsize=14)

# Legend
legend_elements = [
    Patch(facecolor='black', hatch='xx', edgecolor='black', label='Comp (FLARE)'),
    Patch(facecolor='gray', hatch='//', edgecolor='black', label='Comm (FLARE)'),
    Patch(facecolor='dimgray', hatch='..', edgecolor='black', label='Comp (BASELINE)'),
    Patch(facecolor='lightgray', hatch='\\\\', edgecolor='black', label='Comm (BASELINE)'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

plt.tight_layout()
plt.show()
