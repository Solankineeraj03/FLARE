import os, subprocess, numpy as np, time, socket

splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
num_clients = 10
total_data = 50000  # CIFAR-10 train size

BASE = os.path.dirname(os.path.abspath(__file__))
server_py = os.path.join(BASE, "fl_server.py")
client_py = os.path.join(BASE, "fl_client.py")
indices_dir = os.path.join(BASE, "client_indices")
logs_dir = os.path.join(BASE, "logs_msb8_qat")
os.makedirs(indices_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except:
            time.sleep(0.5)
    return False

for pct in splits:
    pct_i = int(pct * 100)
    size = int(total_data * pct)
    per = size // num_clients
    pool = np.random.permutation(total_data)[:size]

    for cid in range(num_clients):
        labeled_idx = pool[cid * per:(cid + 1) * per]
        unlabeled_idx = np.setdiff1d(np.arange(total_data), labeled_idx)
        np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)
        np.save(os.path.join(indices_dir, f"client_{cid}_unlabeled_split_{pct_i}.npy"), unlabeled_idx)

    print(f"📁 Indices saved for {pct_i}%")

    srv_log = open(os.path.join(logs_dir, f"server_{pct_i}.log"), "w", encoding="utf-8")
    srv = subprocess.Popen(["python", server_py, "--rounds", "20"],
                           stdout=srv_log, stderr=subprocess.STDOUT)

    def cleanup_server():
        try:
            srv.terminate()
        except Exception:
            pass
        try:
            srv_log.close()
        except Exception:
            pass

    if not wait_for_server():
        print(f"❌ Server failed for {pct_i}%")
        cleanup_server()
        continue

    procs = []
    for cid in range(num_clients):
        cl_log = open(os.path.join(logs_dir, f"client{cid}_{pct_i}.log"), "w", encoding="utf-8")
        cmd = [
            "python", client_py,
            "--cid", str(cid),
            "--percent", str(pct),
            "--index_dir", indices_dir,
            "--switch_epoch", "1",          # enter full 8-bit early
            "--local_epochs", "6",          # more full-8b time
            "--lr_task", "0.001",
            "--adv_wt", "0.0",
            "--kd_T", "2.0",
        ]
        procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

    for p in procs:
        p.wait()
    cleanup_server()
    print(f"✅ Done split {pct_i}%")



# # # import os

# # # # Define run_experiment_al.py

# # import os
# # import subprocess
# # import numpy as np
# # import time
# # import socket

# # splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# # num_clients = 10
# # total_data  = 50000

# # BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# # server_py   = os.path.join(BASE_DIR, "fl_server.py")
# # client_py   = os.path.join(BASE_DIR, "fl_client.py")
# # indices_dir = os.path.join(BASE_DIR, "client_indices")
# # logs_dir    = os.path.join(BASE_DIR, "logs_alfed_cifar100")
# # os.makedirs(indices_dir, exist_ok=True)
# # os.makedirs(logs_dir, exist_ok=True)

# # def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
# #     start = time.time()
# #     while time.time() - start < timeout:
# #         try:
# #             with socket.create_connection((host, port), timeout=1):
# #                 return True
# #         except:
# #             time.sleep(0.5)
# #     return False

# # for pct in splits:
# #     pct_i = int(pct * 100)
# #     size = int(total_data * pct)
# #     per = size // num_clients
# #     pool = np.random.permutation(total_data)[:size]
    
# #     for cid in range(num_clients):
# #         sub = pool[cid*per:(cid+1)*per]
# #         np.random.shuffle(sub)
# #         half = len(sub) // 2
# #         labeled_idx = sub[:half]
# #         unlabeled_idx = sub[half:]
# #         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)
# #         np.save(os.path.join(indices_dir, f"client_{cid}_unlabeled_split_{pct_i}.npy"), unlabeled_idx)

# #     print(f"📁 Indices saved for {pct_i}%")

# #     srv_log = open(os.path.join(logs_dir, f"server_{pct_i}.log"), "w", encoding="utf-8")
# #     srv_proc = subprocess.Popen(["python", server_py], stdout=srv_log, stderr=subprocess.STDOUT)
# #     if not wait_for_server():
# #         print(f"❌ Server failed for {pct_i}%")
# #         srv_proc.terminate()
# #         srv_log.close()
# #         continue

# #     procs = []
# #     for cid in range(num_clients):
# #         cl_log = open(os.path.join(logs_dir, f"client{cid}_{pct_i}.log"), "w", encoding="utf-8")
# #         cmd = ["python", client_py, "--cid", str(cid), "--percent", str(pct), "--index_dir", indices_dir]
# #         procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

# #     for p in procs:
# #         p.wait()
# #     srv_proc.terminate()
# #     srv_log.close()
# #     print(f"✅ Done split {pct_i}%")

# # run_experiment.py
# import os
# import subprocess
# import numpy as np
# import time
# import socket

# splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# num_clients = 10
# total_data = 50000  # CIFAR-10 train size

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# server_py = os.path.join(BASE_DIR, "fl_server_fedavg.py")
# client_py = os.path.join(BASE_DIR, "fl_client_msb.py")
# indices_dir = os.path.join(BASE_DIR, "client_indices")
# logs_dir = os.path.join(BASE_DIR, "logs_msb_lsb")
# os.makedirs(indices_dir, exist_ok=True)
# os.makedirs(logs_dir, exist_ok=True)

# def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             with socket.create_connection((host, port), timeout=1):
#                 return True
#         except:
#             time.sleep(0.5)
#     return False


# # MSB/LSB parameters forwarded to clients (tweak as needed)
# MSB_BITS = 4
# SWITCH_EPOCH = 5

# for pct in splits:
#     pct_i = int(pct * 100)

#     # ----- build labeled/unlabeled indices per client -----
#     size = int(total_data * pct)
#     per_client = size // num_clients
#     pool = np.random.permutation(total_data)[:size]

#     for cid in range(num_clients):
#         labeled_idx = pool[cid * per_client:(cid + 1) * per_client]
#         unlabeled_idx = np.setdiff1d(np.arange(total_data), labeled_idx)
#         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)
#         np.save(os.path.join(indices_dir, f"client_{cid}_unlabeled_split_{pct_i}.npy"), unlabeled_idx)

#     print(f"📁 Saved indices for split {pct_i}%")

#     # ----- start server -----
#     srv_log_path = os.path.join(logs_dir, f"server_{pct_i}.log")
#     with open(srv_log_path, "w", encoding="utf-8") as srv_log:
#         srv_proc = subprocess.Popen(["python", server_py, "--rounds", "20"],
#                                     stdout=srv_log, stderr=subprocess.STDOUT)

#     if not wait_for_server():
#         print(f"❌ Server failed to start for split {pct_i}%")
#         srv_proc.terminate()
#         continue

#     # ----- start clients -----
#     procs = []
#     for cid in range(num_clients):
#         cl_log_path = os.path.join(logs_dir, f"client{cid}_{pct_i}.log")
#         cl_log = open(cl_log_path, "w", encoding="utf-8")
#         cmd = [
#             "python", client_py,
#             "--cid", str(cid),
#             "--percent", str(pct),
#             "--index_dir", indices_dir,
#             "--msb_bits", str(MSB_BITS),
#             "--switch_epoch", str(SWITCH_EPOCH),
#         ]
#         procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

#     # Wait for all clients
#     for p in procs:
#         p.wait()

#     # Stop server
#     srv_proc.terminate()
#     print(f"✅ Finished split {pct_i}%")


# import os
# import subprocess
# import numpy as np
# import time
# import socket

# splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# num_clients = 10
# total_data = 50000

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# server_py = os.path.join(BASE_DIR, "fl_server.py")
# client_py = os.path.join(BASE_DIR, "fl_client.py")
# indices_dir = os.path.join(BASE_DIR, "client_indices")
# logs_dir = os.path.join(BASE_DIR, "agzf_final2")
# os.makedirs(indices_dir, exist_ok=True)
# os.makedirs(logs_dir, exist_ok=True)

# def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             with socket.create_connection((host, port), timeout=1):
#                 return True
#         except:
#             time.sleep(0.5)
#     return False

# # === Parameters for MSB-LSB logic ===
# MSB_BITS = 4
# SWITCH_EPOCH = 10

# for pct in splits:
#     pct_i = int(pct * 100)
#     size = int(total_data * pct)
#     per = size // num_clients
#     pool = np.random.permutation(total_data)[:size]
    
#     for cid in range(num_clients):
#         labeled_idx = pool[cid * per:(cid + 1) * per]
#         unlabeled_idx = np.setdiff1d(np.arange(total_data), labeled_idx)
    
#         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)
#         np.save(os.path.join(indices_dir, f"client_{cid}_unlabeled_split_{pct_i}.npy"), unlabeled_idx)

#     print(f"📁 Indices saved for {pct_i}%")

#     srv_log = open(os.path.join(logs_dir, f"server_{pct_i}.log"), "w", encoding="utf-8")
#     srv_proc = subprocess.Popen(["python", server_py], stdout=srv_log, stderr=subprocess.STDOUT)
    
#     if not wait_for_server():
#         print(f"❌ Server failed for {pct_i}%")
#         srv_proc.terminate()
#         srv_log.close()
#         continue

#     procs = []
#     for cid in range(num_clients):
#         cl_log = open(os.path.join(logs_dir, f"client{cid}_{pct_i}.log"), "w", encoding="utf-8")
#         cmd = [
#             "python", client_py,
#             "--cid", str(cid),
#             "--percent", str(pct),
#             "--index_dir", indices_dir,
#             # "--msb_bits", str(MSB_BITS),
#             # "--switch_epoch", str(SWITCH_EPOCH)
#         ]
#         procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

#     for p in procs:
#         p.wait()
#     srv_proc.terminate()
#     srv_log.close()
#     print(f"✅ Done split {pct_i}%")
# import os
# import subprocess
# import numpy as np
# import time
# import socket

# splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# num_clients = 10
# total_data  = 50000

# BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# server_py   = os.path.join(BASE_DIR, "fl_server.py")
# client_py   = os.path.join(BASE_DIR, "fl_client.py")
# indices_dir = os.path.join(BASE_DIR, "client_indices")
# logs_dir    = os.path.join(BASE_DIR, "logs_res_c07")
# os.makedirs(indices_dir, exist_ok=True)
# os.makedirs(logs_dir, exist_ok=True)

# def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             with socket.create_connection((host, port), timeout=1):
#                 return True
#         except:
#             time.sleep(0.5)
#     return False

# for pct in splits:
#     pct_i = int(pct * 100)
#     size = int(total_data * pct)
#     per = size // num_clients
#     pool = np.random.permutation(total_data)[:size]
    
#     for cid in range(num_clients):
#         labeled_idx   = pool[cid*per:(cid+1)*per]
#         unlabeled_idx = np.setdiff1d(np.arange(total_data), labeled_idx)
    
#         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)
#         np.save(os.path.join(indices_dir, f"client_{cid}_unlabeled_split_{pct_i}.npy"), unlabeled_idx)

#     print(f"📁 Indices saved for {pct_i}%")

#     srv_log = open(os.path.join(logs_dir, f"server_{pct_i}.log"), "w", encoding="utf-8")
#     srv_proc = subprocess.Popen(["python", server_py], stdout=srv_log, stderr=subprocess.STDOUT)
#     if not wait_for_server():
#         print(f"❌ Server failed for {pct_i}%")
#         srv_proc.terminate()
#         srv_log.close()
#         continue

#     procs = []
#     for cid in range(num_clients):
#         cl_log = open(os.path.join(logs_dir, f"client{cid}_{pct_i}.log"), "w", encoding="utf-8")
#         cmd = ["python", client_py, "--cid", str(cid), "--percent", str(pct), "--index_dir", indices_dir]
#         procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

#     for p in procs:
#         p.wait()
#     srv_proc.terminate()
#     srv_log.close()
#     print(f"✅ Done split {pct_i}%")


# import os
# import subprocess
# import numpy as np
# import time
# import socket

# splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# num_clients = 10
# total_data  = 50000

# BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# server_py   = os.path.join(BASE_DIR, "fl_server.py")
# client_py   = os.path.join(BASE_DIR, "fl_client.py")
# indices_dir = os.path.join(BASE_DIR, "client_indices")
# logs_dir    = os.path.join(BASE_DIR, "logs_res_c07")
# os.makedirs(indices_dir, exist_ok=True)
# os.makedirs(logs_dir, exist_ok=True)

# def wait_for_server(host="127.0.0.1", port=8080, timeout=30):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             with socket.create_connection((host, port), timeout=1):
#                 return True
#         except:
#             time.sleep(0.5)
#     return False

# for pct in splits:
#     pct_i = int(pct * 100)
#     size = int(total_data * pct)
#     per = size // num_clients
#     pool = np.random.permutation(total_data)[:size]
    
#     for cid in range(num_clients):
#         labeled_idx = pool[cid*per:(cid+1)*per]
#         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)

#     print(f"📁 Indices saved for {pct_i}%")

#     srv_log = open(os.path.join(logs_dir, f"server_{pct_i}.log"), "w", encoding="utf-8")
#     srv_proc = subprocess.Popen(["python", server_py], stdout=srv_log, stderr=subprocess.STDOUT)
#     if not wait_for_server():
#         print(f"❌ Server failed for {pct_i}%")
#         srv_proc.terminate()
#         srv_log.close()
#         continue

#     procs = []
#     for cid in range(num_clients):
#         cl_log = open(os.path.join(logs_dir, f"client{cid}_{pct_i}.log"), "w", encoding="utf-8")
#         cmd = ["python", client_py, "--cid", str(cid), "--percent", str(pct), "--index_dir", indices_dir]
#         procs.append(subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT))

#     for p in procs:
#         p.wait()
#     srv_proc.terminate()
#     srv_log.close()
#     print(f"✅ Done split {pct_i}%")
# import os
# import subprocess
# import numpy as np
# import time
# import socket

# splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# num_clients = 10
# total_data = 50000

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# server_py = os.path.join(BASE_DIR, "fl_server.py")
# client_py = os.path.join(BASE_DIR, "fl_client.py")
# indices_dir = os.path.join(BASE_DIR, "client_indices")
# logs_dir = os.path.join(BASE_DIR, "logs_no_al")
# os.makedirs(indices_dir, exist_ok=True)
# os.makedirs(logs_dir, exist_ok=True)

# def wait_for_server(host="131.193.50.219", port=5000, timeout=30):
#     start = time.time()
#     while time.time() - start < timeout:
#         try:
#             with socket.create_connection((host, port), timeout=1):
#                 return True
#         except:
#             time.sleep(0.5)
#     return False

# for pct in splits:
#     pct_i = int(pct * 100)
#     size = int(total_data * pct)
#     per = size // num_clients
#     pool = np.random.permutation(total_data)[:size]

#     for cid in range(num_clients):
#         labeled_idx = pool[cid * per:(cid + 1) * per]
#         np.save(os.path.join(indices_dir, f"client_{cid}_split_{pct_i}.npy"), labeled_idx)

#     print(f"📁 Indices saved for {pct_i}%")

#     # Start server
#     srv_log_path = os.path.join(logs_dir, f"server_{pct_i}.log")
#     with open(srv_log_path, "w", encoding="utf-8") as srv_log:
#         srv_proc = subprocess.Popen(["python", server_py], stdout=srv_log, stderr=subprocess.STDOUT)
#         if not wait_for_server():
#             print(f"❌ Server failed for {pct_i}%")
#             srv_proc.terminate()
#             continue

#         # Start clients and track log files
#         procs = []
#         for cid in range(num_clients):
#             cl_log_path = os.path.join(logs_dir, f"client{cid}_{pct_i}.log")
#             cl_log = open(cl_log_path, "w", encoding="utf-8")
#             cmd = ["python", client_py, "--cid", str(cid), "--percent", str(pct), "--index_dir", indices_dir]
#             p = subprocess.Popen(cmd, stdout=cl_log, stderr=subprocess.STDOUT)
#             procs.append((p, cl_log))  # Save both process and log file handle

#         # Wait for all clients to finish
#         for p, log in procs:
#             p.wait()
#             log.close()

#         srv_proc.terminate()

#     print(f"✅ Done split {pct_i}%")

# import matplotlib.pyplot as plt
# import numpy as np

# # Techniques and their costs (Computation in KB, Communication in KB)
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# comp_without_al = [314.08, 244.29, 279.18, 209.39]
# comp_with_al = [174.91, 139.92, 104.94, 139.92]
# comm_without_al = [4324.44] * 4
# comm_with_al = [4453.33] * 4

# bar_width = 0.35
# x = np.arange(len(techniques))

# # Create stacked bars
# fig, ax = plt.subplots(figsize=(10, 6))

# # Without AL
# bars1 = ax.bar(x - bar_width / 2, comm_without_al, bar_width, label='Comm (w/o AL)', color='#9ecae1')
# bars2 = ax.bar(x - bar_width / 2, comp_without_al, bar_width, bottom=comm_without_al, label='Comp (w/o AL)', color='#3182bd')

# # With AL
# bars3 = ax.bar(x + bar_width / 2, comm_with_al, bar_width, label='Comm (with AL)', color='#fcbba1')
# bars4 = ax.bar(x + bar_width / 2, comp_with_al, bar_width, bottom=comm_with_al, label='Comp (with AL)', color='#de2d26')

# # Labels and aesthetics
# ax.set_ylabel('Cost (KB)')
# ax.set_title('Computation + Communication Costs per FL Method')
# ax.set_xticks(x)
# ax.set_xticklabels(techniques)
# ax.legend()
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Techniques
# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Original values
# comp_without_al = [314.08e9, 244.29e9, 279.18e9, 209.39e9]  # in MACs (billions)
# comp_with_al    = [174.91e9, 139.92e9, 104.94e9, 139.92e9]
# comm_without_al = [4324.44, 4324.44, 4324.44, 4324.44]  # in KB
# comm_with_al    = [4453.33, 4453.33, 4453.33, 4453.33]

# # Option 1: Normalize both to % scale
# max_comp = max(max(comp_without_al), max(comp_with_al))
# max_comm = max(max(comm_without_al), max(comm_with_al))

# comp_wo_norm = [v / max_comp * 100 for v in comp_without_al]
# comp_w_norm  = [v / max_comp * 100 for v in comp_with_al]
# comm_wo_norm = [v / max_comm * 100 for v in comm_without_al]
# comm_w_norm  = [v / max_comm * 100 for v in comm_with_al]

# fig1, ax1 = plt.subplots(figsize=(10, 6))
# ax1.bar(x - bar_width / 2, comm_wo_norm, bar_width, label='Comm (w/o AL)', color='#9ecae1')
# ax1.bar(x - bar_width / 2, comp_wo_norm, bar_width, bottom=comm_wo_norm, label='Comp (w/o AL)', color='#3182bd')
# ax1.bar(x + bar_width / 2, comm_w_norm, bar_width, label='Comm (with AL)', color='#fcbba1')
# ax1.bar(x + bar_width / 2, comp_w_norm, bar_width, bottom=comm_w_norm, label='Comp (with AL)', color='#de2d26')
# ax1.set_ylabel('Normalized Cost (%)')
# ax1.set_title('Option 1: Normalized Cost Comparison')
# ax1.set_xticks(x)
# ax1.set_xticklabels(techniques)
# ax1.legend()

# # Option 2: Twin y-axis
# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax3 = ax2.twinx()
# ax2.bar(x - bar_width / 2, comm_without_al, bar_width, label='Comm (w/o AL)', color='#9ecae1')
# ax2.bar(x + bar_width / 2, comm_with_al, bar_width, label='Comm (with AL)', color='#fcbba1')
# ax3.bar(x - bar_width / 2, comp_without_al, bar_width, label='Comp (w/o AL)', color='#3182bd', alpha=0.5)
# ax3.bar(x + bar_width / 2, comp_with_al, bar_width, label='Comp (with AL)', color='#de2d26', alpha=0.5)
# ax2.set_ylabel('Communication (KB)')
# ax3.set_ylabel('Computation (MACs)')
# ax2.set_title('Option 2: Dual Y-Axis Cost Comparison')
# ax2.set_xticks(x)
# ax2.set_xticklabels(techniques)
# fig2.tight_layout()

# # Option 3: Log scale
# fig3, ax4 = plt.subplots(figsize=(10, 6))
# ax4.bar(x - bar_width / 2, comm_without_al, bar_width, label='Comm (w/o AL)', color='#9ecae1')
# ax4.bar(x - bar_width / 2, comp_without_al, bar_width, bottom=comm_without_al, label='Comp (w/o AL)', color='#3182bd')
# ax4.bar(x + bar_width / 2, comm_with_al, bar_width, label='Comm (with AL)', color='#fcbba1')
# ax4.bar(x + bar_width / 2, comp_with_al, bar_width, bottom=comm_with_al, label='Comp (with AL)', color='#de2d26')
# ax4.set_yscale('log')
# ax4.set_ylabel('Cost (log scale)')
# ax4.set_title('Option 3: Log Scale Cost Comparison')
# ax4.set_xticks(x)
# ax4.set_xticklabels(techniques)
# ax4.legend()

# plt.show()

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
# bar2 = ax1.bar(x + bar_width / 2, comm_with_al, bar_width, color='#fcae91')
# bar3 = ax2.bar(x - bar_width / 2, comp_without_al, bar_width, color='#2171b5', alpha=0.5)
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

# techniques = ["FedAvg", "FedProx", "FedAdam", "FedAdagrad"]
# x = np.arange(len(techniques))
# bar_width = 0.35

# # Scaled values
# comp_with_al = np.array([174.91e9, 139.92e9, 104.94e9, 139.92e9]) / 1e9  # billions of MACs
# comm_with_al = np.array([4453.33] * 4) / 1024  # MB

# fig, ax = plt.subplots(figsize=(10, 5))

# bar1 = ax.bar(x - bar_width/2, comp_with_al, width=bar_width, label="Comp (B MACs)", color="#377eb8")
# bar2 = ax.bar(x + bar_width/2, comm_with_al, width=bar_width, label="Comm (MB)", color="#4daf4a")

# ax.set_ylabel("Cost (B MACs / MB)")
# ax.set_title("Computation vs Communication Cost (with Active Learning)")
# ax.set_xticks(x)
# ax.set_xticklabels(techniques)

# # Add direct labels
# for bar in bar1:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)

# for bar in bar2:
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)

# ax.legend()
# plt.tight_layout()
# plt.show()
