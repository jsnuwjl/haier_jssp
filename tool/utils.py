import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_gantt(logs, path="out/gant.png"):
    logs = pd.DataFrame(logs, columns=["product_id", "route_id", "order", "process", "equ_id", "arrive",
                                       "start", "transport", "end", "start_x", "start_y", "end_x", "end_y"])
    # logs = logs[logs.equ_id <= 25]
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=[8, 6])
    for g_name, g_data in logs.groupby("order", as_index=False):
        ax.fill_between(np.arange(-1, logs["end"].max()) + 1,
                        g_data["equ_id"].min() - 0.5,
                        g_data["equ_id"].max() + 0.5,
                        alpha=0.5)
    logs["job_type"] = LabelEncoder().fit_transform(logs["route_id"])
    c_map = plt.get_cmap('tab10', len(logs["route_id"].unique()))
    for row_id, row in logs.iterrows():
        ax.barh(row["equ_id"],
                width=row["end"] - row["start"],
                left=row["start"],
                edgecolor="black",
                color=c_map(row["job_type"])
                )
        ax.text(x=row["start"],
                y=row["equ_id"] - 0.25,
                s="%d" % (row["product_id"] + 1),
                # s="%s-%02d" % (row["route_id"], row["product_id"] + 1)
                )
    logs["equ_id"] = logs["equ_id"].astype(int)
    ax.set_yticks(np.arange(np.max(logs["equ_id"]) + 1), np.arange(np.max(logs["equ_id"]) + 1) + 1,
                  fontproperties='Times New Roman')
    ax.set_ylabel("Machine")
    ax.set_xlabel("Time")
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("完单时间%.2f" % (logs.end.max()))
