\# FLARE: Battery-Free Federated Learning with Adaptive Resource-Aware Sampling



Implementation of our FLARE framework (intermittent-aware federated active learning with reactive execution).  

Paper: “FLARE: Battery-Free Federated Learning Scheme Using Adaptive Resource-Aware Sampling” (IoT-J submission).



\## Quickstart

1\) Create/activate a Python 3.10+ env  

2\) Install deps: `pip install -r requirements.txt`  

3\) Example: `python main.py --dataset cifar10 --model resnet18 --strategy FedAdam`



\## Repo layout (expected)

\- models/ (Feature Extractor, Classifier, Decoder, Discriminator, Sampler)

\- client/  (local training \& active sampling loop)

\- server/  (FedAvg / FedProx / FedAdam / FedAdagrad aggregation)

\- utils/   (data, logging, evaluation)

\- main.py  (entrypoint)

\- requirements.txt



