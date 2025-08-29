# fl_server_fedavg.py newset one msb lsb 8 bit 
import argparse
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from models import CompVGGFeature as FE, CompVGGClassifier as CL
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    # Initialize server with student weights (FE + Classifier)
    fe, cl = FE(), CL(num_classes=args.num_classes)
    with torch.no_grad():
        init = [*[p.detach().cpu().numpy() for p in fe.parameters()],
                *[p.detach().cpu().numpy() for p in cl.parameters()]]
    strategy = FedAvg(initial_parameters=ndarrays_to_parameters(init))

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()



# # --- fl_server.py ---feddopt
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier



# def get_initial_parameters():
#     model = CompVGGFeature(), CompVGGClassifier(num_classes=10)
#     return ndarrays_to_parameters(
#         [p.detach().cpu().numpy() for m in model for p in m.parameters()]
#     )


# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}


# def main():
#     strategy = fl.server.strategy.FedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         # server_address="0.0.0.0:5000",  # Listen on all interfaces, port 5000[::]:5000
#         server_address="127.0.0.1:8080", 
#         config=fl.server.ServerConfig(num_rounds=11),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()
# --- fl_server.py ---
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import FeatureExtractor, Classifier  # Updated imports

# def get_initial_parameters():
#     # Create instances of the current models
#     fe = FeatureExtractor()
#     clf = Classifier(num_classes=10)  # 10 classes for CIFAR-10
    
#     # Combine parameters from both models
#     params = []
#     params.extend([p.detach().cpu().numpy() for p in fe.parameters()])
#     params.extend([p.detach().cpu().numpy() for p in clf.parameters()])
    
#     return ndarrays_to_parameters(params)
# fl_server.py
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import ResNet18  # Import the same model as client

# def get_initial_parameters():
#     # Create student model same as client
#     net = ResNet18(num_classes=10)
    
#     # Build FE sequential same as client
#     fe = nn.Sequential(
#         net.conv1,
#         net.bn1,
#         nn.ReLU(),
#         net.layer1,
#         net.layer2,
#         net.layer3,
#         net.layer4,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten()
#     )
#     classifier = net.linear
    
#     # Combine parameters in same order as client
#     params = []
#     params.extend([p.detach().cpu().numpy() for p in fe.parameters()])
#     params.extend([p.detach().cpu().numpy() for p in classifier.parameters()])
    
#     return ndarrays_to_parameters(params)

# # ... rest of code unchanged ...

# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}

# def main():
#     strategy = fl.server.strategy.FedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080", 
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()
# --- fl_server.py ---
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# import torch.nn as nn  # Add this import
# from models import ResNet18  # Add this import

# def get_initial_parameters():
#     # Create student model same as client
#     net = ResNet18(num_classes=10)
    
#     # Build FE sequential same as client
#     fe = nn.Sequential(
#         net.conv1,
#         net.bn1,
#         nn.ReLU(),
#         net.layer1,
#         net.layer2,
#         net.layer3,
#         net.layer4,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten()
#     )
#     classifier = net.linear
    
#     # Combine parameters in same order as client
#     params = []
#     params.extend([p.detach().cpu().numpy() for p in fe.parameters()])
#     params.extend([p.detach().cpu().numpy() for p in classifier.parameters()])
    
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}

# def main():
#     strategy = fl.server.strategy.FedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080", 
#         config=fl.server.ServerConfig(num_rounds=10),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()
# for active learning 
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch

# from models import CompVGGFeature, CompVGGClassifier, Discriminator

# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=100)
#     disc = Discriminator()

#     # Only send FeatureExtractor + Classifier params to clients
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total > 0 else 0.0}

# def main():
#     strategy = fl.server.strategy.FedAdam(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=10,
#         min_evaluate_clients=10,
#         min_available_clients=10,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=2),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()
# import flwr as fl
# import argparse
# import numpy as np
# from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
# from typing import List, Tuple, Dict, Optional

# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator
# from flwr.server.strategy import FedAvg

# def get_initial_parameters():
#     model_parts = (
#         CompVGGFeature(),
#         CompVGGClassifier(num_classes=10),
#         Decoder(),
#         Discriminator()
#     )
#     params = [p.detach().cpu().numpy() for model in model_parts for p in model.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}

# def log_param_stats(parameters, round_num):
#     param_list = parameters_to_ndarrays(parameters)
#     total_bytes = sum(p.nbytes for p in param_list)
#     print(f"[Round {round_num}] Aggregated Parameter Size: {total_bytes / 1024:.2f} KB")

# class MSBAwareFedAvg(FedAvg):
#     def __init__(self, switch_epoch=5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.switch_epoch = switch_epoch
#         self.current_params = kwargs.get("initial_parameters", None)

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[fl.common.Parameters, Dict[str, float]]]:
        
#         if not results:
#             print(f"[Round {rnd}] No successful results, returning last known parameters.")
#             return self.current_params, {}

#         # Call parent aggregation to get standard FedAvg results
#         aggregated_result = super().aggregate_fit(rnd, results, failures)
        
#         if aggregated_result is None:
#             return self.current_params, {}
            
#         aggregated_params, metrics = aggregated_result
#         self.current_params = aggregated_params

#         # Log parameter statistics
#         log_param_stats(aggregated_params, rnd)
#         print(f"Aggregated Round {rnd} | {'MSB-Only' if rnd < self.switch_epoch else 'Full LSB Update'}")
        
#         return aggregated_params, metrics

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=15)
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     args = parser.parse_args()

#     strategy = MSBAwareFedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         switch_epoch=args.switch_epoch,
#     )

#     # Revert to using start_server() since start_superlink() isn't available
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()
# fl_server.py
# fl_server.py
# import argparse
# from typing import List, Tuple, Dict, Optional

# import numpy as np
# import flwr as fl
# from flwr.common import (
#     Parameters,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )
# from flwr.server.strategy import FedAvg

# from models import CompVGGFeature, CompVGGClassifier


# class SparseMSBFedAvg(FedAvg):
#     """
#     FedAvg variant that transmits sparse MSB deltas for the first
#     `switch_epoch` rounds and falls back to dense residuals afterwards.
#     """

#     def __init__(
#         self,
#         *,
#         switch_epoch: int,
#         msb_bits: int,
#         initial_parameters: Parameters,
#         **kwargs,
#     ):
#         super().__init__(initial_parameters=initial_parameters, **kwargs)

#         self.switch_epoch = switch_epoch
#         self.msb_bits = msb_bits
#         self.step = 2 ** -msb_bits  # quantisation step

#         # Shapes of every parameter tensor (FE + Classifier only)
#         init_arrays = parameters_to_ndarrays(initial_parameters)
#         self.shapes = [arr.shape for arr in init_arrays]

#         # Snapshot of the previous global int-MSB array per layer
#         self.prev_int: List[Optional[np.ndarray]] = [None] * len(self.shapes)

#     # ------------------------------------------------------------------ #
#     # FedAvg hook                                                         #
#     # ------------------------------------------------------------------ #
#     def aggregate_fit(  # type: ignore[override]
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, float]]]:

#         if not results:  # no client updates this round
#             return self.initial_parameters, {}

#         # ---------------- Phase 1: sparse-MSB averaging ---------------- #
#         if rnd < self.switch_epoch:
#             client_msbs: List[List[np.ndarray]] = []

#             for _, fit_res in results:
#                 flat = parameters_to_ndarrays(fit_res.parameters)

#                 # every layer must arrive as (idxs, vals)
#                 if len(flat) != 2 * len(self.shapes):
#                     raise ValueError(
#                         f"Expected {2*len(self.shapes)} arrays, got {len(flat)}"
#                     )

#                 layer_msbs: List[np.ndarray] = []
#                 for layer_idx in range(len(self.shapes)):
#                     idxs = flat[2 * layer_idx].astype(np.int32).ravel()
#                     vals = flat[2 * layer_idx + 1].astype(np.int16).ravel()

#                     if len(idxs) != len(vals):
#                         raise ValueError(
#                             f"Index/value mismatch from client in layer {layer_idx}: "
#                             f"{len(idxs)} vs {len(vals)}"
#                         )

#                     shape = self.shapes[layer_idx]
#                     total = int(np.prod(shape))

#                     int_buf = np.zeros(total, dtype=np.int16)
#                     if idxs.size:
#                         int_buf[idxs] = vals

#                     msb = int_buf.astype(np.float32).reshape(shape) * self.step
#                     layer_msbs.append(msb)

#                 client_msbs.append(layer_msbs)

#             # average MSBs layer-by-layer
#             agg_msbs: List[np.ndarray] = []
#             for layer_idx, shape in enumerate(self.shapes):
#                 stack = np.stack([c[layer_idx] for c in client_msbs], axis=0)
#                 agg_msbs.append(stack.mean(axis=0))

#             # re-quantise, delta-encode, pack sparse
#             payload: List[np.ndarray] = []
#             for layer_idx, arr in enumerate(agg_msbs):
#                 int_arr = np.round(arr / self.step).astype(np.int16).ravel()
#                 prev_int_arr = self.prev_int[layer_idx]

#                 changed = (
#                     np.ones_like(int_arr, dtype=bool)
#                     if prev_int_arr is None
#                     else int_arr != prev_int_arr
#                 )

#                 idxs = np.nonzero(changed)[0].astype(np.int32)
#                 vals = int_arr[changed].astype(np.int16)

#                 payload.extend([idxs, vals])
#                 self.prev_int[layer_idx] = int_arr.copy()

#             return ndarrays_to_parameters(payload), {}

#         # ------------- Phase 2: dense residual averaging ---------------- #
#         # Some clients might still send sparse arrays: convert them to
#         # *zero residual* tensors so shapes match.
#         dense_results = []
#         for proxy, fit_res in results:
#             flat = parameters_to_ndarrays(fit_res.parameters)

#             is_dense = (
#                 len(flat) == len(self.shapes) and flat[0].dtype.kind == "f"
#             )
#             if is_dense:
#                 dense_results.append((proxy, fit_res))
#                 continue

#             # Convert sparse → dense zeros (ignores vals, keeps shape)
#             dense = [np.zeros(shape, dtype=np.float32) for shape in self.shapes]
#             fit_res.parameters = ndarrays_to_parameters(dense)
#             dense_results.append((proxy, fit_res))

#         aggregated = super().aggregate_fit(rnd, dense_results, failures)
#         if aggregated is None:
#             return None

#         params, metrics = aggregated
#         return params, metrics


# # ---------------------------------------------------------------------- #
# # Helpers                                                                #
# # ---------------------------------------------------------------------- #
# def get_initial_parameters() -> Parameters:
#     """Return FeatureExtractor + Classifier weights as initial global model."""
#     parts = (CompVGGFeature(), CompVGGClassifier(num_classes=10))
#     arrays = [p.detach().cpu().numpy() for m in parts for p in m.parameters()]
#     return ndarrays_to_parameters(arrays)


# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     if total == 0:
#         return {"accuracy": 0.0}
#     acc = sum(n * m["accuracy"] for n, m in metrics) / total
#     return {"accuracy": acc}


# # ---------------------------------------------------------------------- #
# # Entry point                                                            #
# # ---------------------------------------------------------------------- #
# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=20)
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     strategy = SparseMSBFedAvg(
#         switch_epoch=args.switch_epoch,
#         msb_bits=args.msb_bits,
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     # (The API is marked deprecated but still works; swap to flower-superlink later.)
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=args.rounds),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()
# -----------------------------------------------------------
#  Minimal Flower server that understands AGZF sparse payload
# -----------------------------------------------------------
# fl_server.py  –  AGZF-aware Flower server with log of active indices
# --------------------------------------------------------------------
# ------------------------------------------------------------
# fl_server.py  –  Flower server logging active indices / round
# ------------------------------------------------------------
# ------------------------------------------------------------
# fl_server.py  –  Flower server with AGZF and proper
#                  unique-index logging (no double-counting)
# fl_server_fedavg.py
# import argparse
# import flwr as fl
# from flwr.server.strategy import FedAvg
# from flwr.common import ndarrays_to_parameters
# import numpy as np
# import torch

# from models import CompVGGFeature as FE, CompVGGClassifier as CL


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=20)
#     parser.add_argument("--num_classes", type=int, default=10)
#     args = parser.parse_args()

#     # Initialize student weights (FE + Classifier) as the server's starting point
#     dummy_FE, dummy_CL = FE(), CL(num_classes=args.num_classes)
#     initial_weights = [
#         *[p.detach().cpu().numpy() for p in dummy_FE.parameters()],
#         *[p.detach().cpu().numpy() for p in dummy_CL.parameters()],
#     ]
#     initial_parameters = ndarrays_to_parameters(initial_weights)

#     strategy = FedAvg(initial_parameters=initial_parameters)

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=args.rounds),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()

# Updated Flower server for global-round-based freezing with AGZF-aware clients
# import flwr as fl
# import argparse
# import numpy as np
# from typing import List, Tuple
# from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
# from flwr.server.strategy import FedAvg


# class AGZFFedAvg(FedAvg):
#     """
#     FedAvg strategy that handles sparse parameter uploads from AGZF clients
#     and performs dense aggregation over all parameters (frozen + active).
#     """
#     def __init__(self, initial_parameters: Parameters, **kwargs):
#         super().__init__(initial_parameters=initial_parameters, **kwargs)
#         self.full_weights = parameters_to_ndarrays(initial_parameters)  # keep full param state

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Tuple[Parameters, dict]:

#         # Collect sparse updates
#         total_examples = sum([r.num_examples for _, r in results])
#         delta = [np.zeros_like(w) for w in self.full_weights]

#         for _, fit_res in results:
#             sparse_weights = parameters_to_ndarrays(fit_res.parameters)
#             offset = 0
#             for i, full_tensor in enumerate(self.full_weights):
#                 numel = full_tensor.size
#                 idxs = sparse_weights[offset].astype(np.int32)
#                 vals = sparse_weights[offset + 1].astype(np.float32)
#                 offset += 2

#                 # Accumulate update weighted by number of examples
#                 local_delta = np.zeros_like(full_tensor)
#                 local_delta.flat[idxs] = vals - self.full_weights[i].flat[idxs]
#                 delta[i] += local_delta * (fit_res.num_examples / total_examples)

#         # Apply aggregated delta to full weights
#         self.full_weights = [w + d for w, d in zip(self.full_weights, delta)]
#         return ndarrays_to_parameters(self.full_weights), {}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=20)
#     args = parser.parse_args()

#     # Load initial model weights
#     import torch
#     from models import CompVGGFeature as FE, CompVGGClassifier as CL

#     dummy_FE, dummy_CL = FE(), CL(num_classes=10)
#     initial_weights = [
#         *[p.detach().numpy() for p in dummy_FE.parameters()],
#         *[p.detach().numpy() for p in dummy_CL.parameters()],
#     ]
#     initial_parameters = ndarrays_to_parameters(initial_weights)

#     strategy = AGZFFedAvg(initial_parameters=initial_parameters)

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=args.rounds),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()

# ------------------------------------------------------------
# import argparse
# from typing import List, Tuple

# import flwr as fl
# import numpy as np
# from flwr.common import (
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
#     Parameters,
# )
# from flwr.server.strategy import FedAvg

# from models import CompVGGFeature, CompVGGClassifier

# # ------------------------------------------------------------------ #
# # 1.  build initial global parameters                                #
# # ------------------------------------------------------------------ #
# def initial_parameters() -> Parameters:
#     nets = (CompVGGFeature(), CompVGGClassifier(num_classes=10))
#     arrs = [p.detach().cpu().numpy() for m in nets for p in m.parameters()]
#     return ndarrays_to_parameters(arrs)

# # ------------------------------------------------------------------ #
# # 2.  FedAvg variant with AGZF-style sparse payloads                 #
# # ------------------------------------------------------------------ #
# class AGZFFedAvg(FedAvg):
#     def __init__(self, *, init: Parameters, **kwargs):
#         super().__init__(initial_parameters=init, **kwargs)
#         self.global_weights = parameters_to_ndarrays(init)  # list[np.ndarray]

#     # -------------------------------------------------------------- #
#     def aggregate_fit(  # type: ignore[override]
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures,
#     ):
#         if not results:
#             return self.initial_parameters, {}

#         # copy current global model so we can overwrite active weights
#         agg = [w.copy() for w in self.global_weights]

#         # -------- collect unique indices from all clients ---------- #
#         layer_union: List[set[int]] = [set() for _ in agg]

#         for _, fit in results:
#             arrs = parameters_to_ndarrays(fit.parameters)  # [idx,val,idx,val,…]
#             for li, (idxs, vals) in enumerate(zip(arrs[::2], arrs[1::2])):
#                 # add to union set (deduplicates automatically)
#                 layer_union[li].update(int(i) for i in idxs)

#                 if idxs.size:  # overwrite active positions
#                     flat = agg[li].ravel()
#                     flat[idxs.astype(np.int64)] = vals

#         # -------- single log line per layer (unique counts) -------- #
#         for li, (union_set, w) in enumerate(zip(layer_union, agg)):
#             act = len(union_set)
#             pct = 100 * act / w.size if w.size else 0
#             print(f"[Server] round {rnd:02d}  layer {li:02d}: "
#                   f"active {act:7d}/{w.size:7d}  ({pct:5.2f} %)"
#             )

#         self.global_weights = agg
#         return ndarrays_to_parameters(agg), {}

# # ------------------------------------------------------------------ #
# # 3.  optional weighted-average accuracy metric                      #
# # ------------------------------------------------------------------ #
# def weighted_avg(metrics):
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics) / total if total else 0.0
#     return {"accuracy": acc}

# # ------------------------------------------------------------------ #
# # 4.  entry-point                                                    #
# # ------------------------------------------------------------------ #
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=20)
#     args = parser.parse_args()

#     strategy = AGZFFedAvg(
#         init=initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_avg,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     # NOTE: start_server() is deprecated but still works;
#     #       switch to flower-superlink when you update Flower.
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=args.rounds),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()



# Updated fl_server.py with MSB-LSB aware aggregation
# import flwr as fl
# import argparse
# import numpy as np
# from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
# from typing import List, Tuple, Dict, Optional

# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator
# from flwr.server.strategy import FedAvg

# def get_initial_parameters():
#     model_parts = (
#         CompVGGFeature(),
#         CompVGGClassifier(num_classes=10),
#         Decoder(),
#         Discriminator()
#     )
#     params = [p.detach().cpu().numpy() for model in model_parts for p in model.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}

# def log_param_stats(parameters, round_num):
#     param_list = parameters_to_ndarrays(parameters)
#     total_bytes = sum(p.nbytes for p in param_list)
#     print(f"[Round {round_num}] Aggregated Parameter Size: {total_bytes / 1024:.2f} KB")

# class MSBAwareFedAvg(FedAvg):
#     def __init__(self, switch_epoch=5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.switch_epoch = switch_epoch
#         self.current_params = kwargs.get("initial_parameters", None)

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[fl.common.Parameters, Dict[str, float]]]:
        
#         if not results:
#             print(f"[Round {rnd}] No successful results, returning last known parameters.")
#             return self.current_params, {}

#         # Call parent aggregation to get standard FedAvg results
#         aggregated_result = super().aggregate_fit(rnd, results, failures)
        
#         if aggregated_result is None:
#             return self.current_params, {}
            
#         aggregated_params, metrics = aggregated_result
#         self.current_params = aggregated_params

#         # Log parameter statistics
#         log_param_stats(aggregated_params, rnd)
#         print(f"Aggregated Round {rnd} | {'MSB-Only' if rnd < self.switch_epoch else 'Full LSB Update'}")
        
#         return aggregated_params, metrics

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=15)
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     args = parser.parse_args()

#     strategy = MSBAwareFedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         switch_epoch=args.switch_epoch,
#     )

#     # Revert to using start_server() since start_superlink() isn't available
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()

# import argparse
# import numpy as np
# import flwr as fl
# from typing import List, Tuple, Dict, Optional
# from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters
# from flwr.server.strategy import FedAvg

# # Only federate FeatureExtractor + Classifier:
# from models import CompVGGFeature, CompVGGClassifier


# def get_initial_parameters() -> Parameters:
#     """Return initial FE+Classifier weights (as float32)."""
#     fe = CompVGGFeature()
#     clf = CompVGGClassifier(num_classes=10)
#     arrays = [p.detach().cpu().numpy() for p in list(fe.parameters()) + list(clf.parameters())]
#     return ndarrays_to_parameters(arrays)


# def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
#     total = sum(n for n, _ in metrics)
#     if total == 0:
#         return {"accuracy": 0.0}
#     return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total}


# class MSBLSBServer(FedAvg):
#     def __init__(
#         self,
#         switch_epoch: int,
#         msb_bits: int,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.switch_epoch = switch_epoch
#         self.msb_bits = msb_bits
#         self.msb_store: Optional[List[np.ndarray]] = None

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[Tuple[Parameters, Dict[str, float]]:
#         # 1) Standard FedAvg aggregation
#         parent = super().aggregate_fit(rnd, results, failures)
#         if parent is None:
#             return None

#         params_aggregated, metrics = parent
#         arrays = parameters_to_ndarrays(params_aggregated)
#         step = 2 ** -self.msb_bits

#     # 2) MSB‐only phase
#         if rnd < self.switch_epoch:
#             # Convert to float32 weights
#             msb_arrays = [
#                 (arr.astype(np.float32) * step)
#                 for arr in arrays
#             ]
#             self.msb_store = [w.copy() for w in msb_arrays]
        
#         # Quantize back to int16 for broadcasting
#             broadcast_arrays = [
#                 np.round(w / step).astype(np.int16)
#                 for w in msb_arrays
#             ]
#             size_kb = sum(w.nbytes for w in broadcast_arrays) / 1024.0
#             print(f"[Round {rnd}] Broadcasting MSB-only ({size_kb:.2f} KB)")
#             return ndarrays_to_parameters(broadcast_arrays), metrics

#     # 3) LSB phase
#     if self.msb_store is None:
#         print(f"[Round {rnd}] ⚠️ No MSB base, broadcasting raw floats")
#         size_kb = sum(a.nbytes for a in arrays) / 1024.0
#         return ndarrays_to_parameters(arrays), metrics

#     # Reconstruct full weights
#     full = [
#         msb + arr
#         for msb, arr in zip(self.msb_store, arrays)
#     ]
#     self.msb_store = [w.copy() for w in full]
#     size_kb = sum(w.nbytes for w in full) / 1024.0
#     print(f"[Round {rnd}] Broadcasting full weights ({size_kb:.2f} KB)")
#     return ndarrays_to_parameters(full), metrics


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rounds", type=int, default=20)
#     parser.add_argument("--switch_epoch", type=int, default=5)
#     parser.add_argument("--msb_bits", type=int, default=6)
#     args = parser.parse_args()

#     strategy = MSBLSBServer(
#         switch_epoch=args.switch_epoch,
#         msb_bits=args.msb_bits,
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()



# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Decoder, Discriminator
# from flwr.server.strategy import FedAvg


# def get_initial_parameters():
#     model = CompVGGFeature(), CompVGGClassifier(num_classes=10), Decoder(), Discriminator()
#     return ndarrays_to_parameters(
#         [p.detach().cpu().numpy() for m in model for p in m.parameters()]
#     )


# def weighted_average(metrics):
#     total = sum([num_examples for num_examples, _ in metrics])
#     acc_sum = sum([num_examples * m["accuracy"] for num_examples, m in metrics])
#     return {"accuracy": acc_sum / total if total > 0 else 0.0}


# def main():
#     strategy = fl.server.strategy.FedAvg(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()


# fl_server.py for fedavg without al
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier
# import sys, io

# # 1) Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# def get_initial_parameters():
#     # Instantiate fresh student: feature extractor + classifier
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     # Collect parameters in the same order as the client get_parameters()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     return ndarrays_to_parameters(params)


# def weighted_average(metrics):
#     # Aggregate client‐side test accuracies
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}


# class FedAvgServer(fl.server.strategy.FedAvg):
#     def evaluate(self, rnd, parameters):
#         # Called at the end of each round to aggregate test metrics
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result


# if __name__ == "__main__":
#     strategy = FedAvgServer(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )
# fedavg plus al 
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Discriminator
# import sys, io

# # 1) Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# def get_initial_parameters():
#     # Initialize student: feature extractor + classifier + discriminator
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     disc = Discriminator()
#     # Collect parameters in the same order as the client get_parameters()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     params += [p.cpu().detach().numpy() for p in disc.parameters()]
#     return ndarrays_to_parameters(params)


# def weighted_average(metrics):
#     # Aggregate client-side test accuracies
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}


# class FedAvgServer(fl.server.strategy.FedAvg):
#     def evaluate(self, rnd, parameters):
#         # Called at the end of each round to aggregate test metrics
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result


# if __name__ == "__main__":
#     strategy = FedAvgServer(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier
# import sys, io

# # Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc   = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}

# class FedProxServer(fl.server.strategy.FedProx):
#     def evaluate(self, rnd, parameters):
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result

# if __name__ == "__main__":
#     strategy = FedProxServer(
#         proximal_mu=0.1,  # Enable FedProx regularization
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )


# # --- FedProx Server WITH Active Learning ---
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Discriminator
# import sys, io

# # Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     disc = Discriminator()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     params += [p.cpu().detach().numpy() for p in disc.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc   = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}

# class FedProxServer(fl.server.strategy.FedProx):
#     def evaluate(self, rnd, parameters):
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result

# if __name__ == "__main__":
#     strategy = FedProxServer(
#         proximal_mu=0.1,  # Enable FedProx regularization
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )
# --- FedProx Server WITH Active Learning and Cost Logging ---
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Discriminator
# import sys, io

# # Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# # === Cost Constants ===
# EPOCHS = 5                # Local epochs
# DATA_PER_CLIENT = 2500   # Avg. CIFAR-10 samples per client
# MACS_PER_SAMPLE = 2e6    # MACs/sample (~CompVGGFeature + Classifier)
# PARAMS = 553514 + 1327872 + 262657  # Total params for FE + Classifier + Discriminator
# BYTES_PER_PARAM = 4       # FP32

# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     disc = Discriminator()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     params += [p.cpu().detach().numpy() for p in disc.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}

# class FedProxServer(fl.server.strategy.FedProx):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.cost_log = []

#     def aggregate_fit(self, rnd, results, failures):
#         num_clients = len(results)

#         # === Cost Estimation ===
#         computation_cost = rnd * num_clients * EPOCHS * DATA_PER_CLIENT * MACS_PER_SAMPLE
#         communication_cost = rnd * num_clients * 2 * PARAMS * BYTES_PER_PARAM

#         print(f"\n[Round {rnd}] Participating Clients: {num_clients}")
#         print(f"[Round {rnd}] Computation Cost: {computation_cost:.2e} MACs")
#         print(f"[Round {rnd}] Communication Cost: {communication_cost / 1e6:.2f} MB")

#         # Store metrics
#         self.cost_log.append({
#             "round": rnd,
#             "clients": num_clients,
#             "computation_MACs": computation_cost,
#             "communication_MB": communication_cost / 1e6,
#         })

#         return super().aggregate_fit(rnd, results, failures)

#     def evaluate(self, rnd, parameters):
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result

# if __name__ == "__main__":
#     strategy = FedProxServer(
#         proximal_mu=0.1,
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#     )
#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )


# fl_server.py for fedadam without al
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier
# import sys, io

# # 1) Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# def get_initial_parameters():
#     # Instantiate fresh student: feature extractor + classifier
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     return ndarrays_to_parameters(params)


# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}


# if __name__ == "__main__":
#     # Replace FedAvg with FedAdam
#     strategy = fl.server.strategy.FedAdam(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         # how many clients must fit/evaluate per round
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         # FedAdam hyperparameters:
#         eta=0.01,        # server learning rate
#         beta_1=0.9,     # momentum term
#         beta_2=0.999,   # second-moment term
#         tau=1e-6,       # regularization
#         accept_failures=True,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )
# fl_server_fedadam_al.py
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Discriminator
# import sys, io

# # Force UTF-8 printing on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# def get_initial_parameters():
#     # Feature Extractor, Classifier, Discriminator for AL setup
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     disc = Discriminator()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     params += [p.cpu().detach().numpy() for p in disc.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}

# class FedAdamServer(fl.server.strategy.FedAdam):
#     def evaluate(self, rnd, parameters):
#         print(f"[Server]   Evaluating round {rnd}…")
#         result = super().evaluate(rnd, parameters)
#         if result is not None:
#             loss, metrics = result
#             print(f"[Server]  Global test accuracy after round {rnd}: {metrics['accuracy']*100:.2f}%")
#         return result

# if __name__ == "__main__":
#     strategy = FedAdamServer(
#         initial_parameters=get_initial_parameters(),
#         evaluate_metrics_aggregation_fn=weighted_average,
#         min_fit_clients=3,
#         min_evaluate_clients=3,
#         min_available_clients=3,
#         # FedAdam hyperparameters:
#         eta=0.01,
#         beta_1=0.9,
#         beta_2=0.999,
#         tau=1e-6,
#         accept_failures=True,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=30),
#         strategy=strategy,
#     )

# # fl_server.py for fedadagrad without al
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier
# import sys, io

# # Force UTF-8 on Windows consoles
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     return ndarrays_to_parameters(params)


# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc   = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}


# if __name__ == "__main__":
#     strategy = fl.server.strategy.FedAdagrad(
#         initial_parameters               = get_initial_parameters(),
#         evaluate_metrics_aggregation_fn = weighted_average,
#         min_fit_clients                  = 3,
#         min_evaluate_clients             = 3,
#         min_available_clients           = 3,
#         eta                              = 0.01,   # Global learning rate
#         eta_l                            = 0.1,    # Local learning rate on client
#         tau                              = 1e-6,   # Small value to prevent division by 0
#         accept_failures                  = True,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )
# # fedadagrad with al
# import flwr as fl
# from flwr.common import ndarrays_to_parameters
# import torch
# from models import CompVGGFeature, CompVGGClassifier, Discriminator
# import sys, io

# # Force UTF-8 output on Windows
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# def get_initial_parameters():
#     fe = CompVGGFeature()
#     cl = CompVGGClassifier(num_classes=10)
#     disc = Discriminator()
#     params = [p.cpu().detach().numpy() for p in fe.parameters()]
#     params += [p.cpu().detach().numpy() for p in cl.parameters()]
#     params += [p.cpu().detach().numpy() for p in disc.parameters()]
#     return ndarrays_to_parameters(params)

# def weighted_average(metrics):
#     total = sum(n for n, _ in metrics)
#     acc   = sum(n * m["accuracy"] for n, m in metrics)
#     return {"accuracy": acc / total if total else 0.0}

# if __name__ == "__main__":
#     strategy = fl.server.strategy.FedAdagrad(
#         initial_parameters               = get_initial_parameters(),
#         evaluate_metrics_aggregation_fn = weighted_average,
#         min_fit_clients                  = 3,
#         min_evaluate_clients             = 3,
#         min_available_clients            = 3,
#         eta                              = 0.01,   # Server LR
#         eta_l                            = 0.1,    # Client LR (if used)
#         tau                              = 1e-6,   # Stability constant
#         accept_failures                  = True,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8080",
#         config=fl.server.ServerConfig(num_rounds=15),
#         strategy=strategy,
#     )
