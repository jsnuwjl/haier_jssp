from tool.env import *
from tool.dqn import DQN
import torch
import numpy as np
import os
import datetime
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def dqn_infer():
    np.random.seed(1)
    env_and_agent = torch.load('out/env_and_agent.pt')
    config = env_and_agent['config']
    config["device"] = 'cpu'
    dqn = DQN(config)
    dqn.eval_net.load_state_dict(env_and_agent['state_dict'])
    js = JobShop()
    job, equ, s = js.reset()
    start_time = datetime.datetime.now()
    while True:
        a = dqn.choose_action(s)
        r, s_, job, equ, done = js.step(job, equ, a)
        dqn.store_transition(s, a, r, s_, done)
        s = s_
        if done:
            break
    end_time = datetime.datetime.now()
    td = end_time - start_time
    print("计算耗时%d.%ss" % (td.seconds, str(td.microseconds)[:2]))
    plot_gantt(js.logs, path="out/dqn.png")


def spt_infer():
    np.random.seed(1)
    js = JobShop()
    job, equ, s = js.reset()
    start_time = datetime.datetime.now()
    while True:
        r, s_, job, equ, done = js.step(job, equ, np.random.randint(0, 5))
        if done:
            break
    end_time = datetime.datetime.now()
    td = end_time - start_time
    print("计算耗时%d.%ss" % (td.seconds, str(td.microseconds)[:2]))
    plot_gantt(js.logs, path="out/spt.png")


dqn_infer()
spt_infer()
