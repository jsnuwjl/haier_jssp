import pandas as pd
import numpy as np
from .utils import *


def masked_argmin(x, condition):
    valid_idx = np.where(condition)[0]
    return valid_idx[x[valid_idx].argmin()]


class Equ:
    def __init__(self):
        set_seed(1)
        self.spt = np.array([6.21, 6.95, 6.72, 7.07, 6.69])
        self.num = np.round(self.spt / np.sum(self.spt) * 25).astype(int)
        self.id = np.arange(np.sum(self.num))
        self.type = np.repeat(range(len(self.spt)), self.num)
        self.spt = np.repeat(self.spt, self.num) * (np.random.random(25) * 0.05 + 0.975)
        self.x = np.hstack([range(1, 6) for _ in range(5)])
        self.y = np.repeat(range(1, 6), 5)
        self.end = np.zeros_like(self.id).astype(float)
        self.total = np.zeros_like(self.id).astype(float)
        self.first_start = np.zeros_like(self.id)


class JobShop:
    def __init__(self, n_class=3):
        self.n_class = n_class
        self.num = np.random.randint(10, 20, n_class)
        self.id = np.arange(3)
        self.name = ['卫衣', 'T恤', 'POLO']
        self.route_num = np.random.randint(4, 6, 3)
        self.dtl = {x: np.sort(np.random.choice(range(5), y, replace=False)) for x, y in
                    zip(self.name, self.route_num)}
        self.spt = np.array([4.21, 6.95, 13.72, 7.07, 19.69])
        self.logs = []

    def reset(self):
        job = Job(self)
        equ = Equ()
        s = self.state(job, equ)
        return job, equ, s

    def state(self, job, equ):
        # 系统中工件数量与总工件数量之比
        x1 = len(self.logs) / np.sum(job.max)
        # 设备利用率
        x2 = np.mean(equ.total > 0)
        # 表示系统中候选工件最小开始加工时间归一化公式
        x3 = job.arrive.min() / (job.arrive.max() + 0.001)
        # 表示系统内候选工件最小延迟归一化公式
        delay_time = job.expected - job.total - job.arrive
        x4 = delay_time.min() / (delay_time.max() + 0.001)
        # reward
        x5 = np.mean(equ.total) / (np.max(equ.end) + 0.001)
        return [x1, x2, x3, x4, x5]

    def step(self, job, equ, action=np.random.randint(0, 1)):
        job.process = list(
            map(lambda x, y: self.spt[self.dtl[self.name[x]][y]] if y < len(self.dtl[self.name[x]]) else np.inf,
                job.type, job.now))
        job.process = np.array(job.process)
        job.start = np.where(job.arrive > 0, job.arrive, 0)
        job.end = job.process + job.start
        if action == 0:
            # FIFO 待加工工件越早达到越优先
            job_id = masked_argmin(job.arrive, ~np.isinf(job.process))
        elif action == 1:
            # SPT 待加工工件加工时间越短越优先
            job_id = masked_argmin(job.process, ~np.isinf(job.process))
        elif action == 2:
            # LWR 待加工工件剩余加工时间越短越优先
            job_id = masked_argmin(job.remain, ~np.isinf(job.process))
        elif action == 3:
            # EDD 待加工工件交货期最早的工件优先
            job_id = masked_argmin(job.expected, ~np.isinf(job.process))
        else:
            # SRPT 待加工工件剩余处理时间越短越优先
            job_id = masked_argmin(np.array(job.remain) / np.array(job.total), ~np.isinf(job.process))
        equ.transport = np.abs(equ.x - 2.5) + np.abs(job.x[job_id] - 2.5) + np.abs(equ.y - job.y[job_id])
        equ.arrive = equ.transport + job.start[job_id]
        equ.start = np.where(equ.end > equ.arrive, equ.end, equ.arrive)
        equ_id = masked_argmin(equ.start + equ.spt + equ.total * 0.01,
                               equ.type == self.dtl[self.name[job.type[job_id]]][job.now[job_id]])
        equ.end[equ_id] = equ.start[equ_id] + equ.spt[equ_id]
        self.logs.append([job.id[job_id], self.name[job.type[job_id]], job.now[job_id], equ.spt[equ_id],
                          equ.id[equ_id], equ.arrive[equ_id], equ.start[equ_id], equ.transport[equ_id],
                          equ.end[equ_id], job.x[job_id], job.y[job_id], equ.x[equ_id], equ.y[equ_id],
                          ])
        job.now[job_id] += 1
        job.arrive[job_id] = equ.end[equ_id]
        job.x[job_id] = equ.x[equ_id]
        job.y[job_id] = equ.y[equ_id]
        equ.total[equ_id] += equ.spt[equ_id]
        job.remain = np.array([np.sum(self.spt[self.dtl[self.name[x]][y:]]) for x, y in zip(job.type, job.now)])
        r = np.mean(equ.total) / (np.max(equ.end) + 0.001)
        s = self.state(job, equ)
        return r, s, job, equ, s[0] == 1


class Job:
    def __init__(self, env: JobShop):
        self.id = np.arange(np.sum(env.num))
        self.arrive = np.repeat(0, np.sum(env.num))  # np.repeat(np.random.randint(0, 10, env.n_class), env.num)
        self.expected = np.repeat(np.random.randint(200, 300, env.n_class), env.num)
        self.type = np.repeat(env.id, env.num)
        self.now = np.repeat(0, np.sum(env.num))
        self.x = np.repeat(2.5, np.sum(env.num))
        self.y = np.repeat(0, np.sum(env.num))
        self.total = np.array([np.sum(env.spt[env.dtl[env.name[x]]]) for x in self.type])
        self.remain = np.array([np.sum(env.spt[env.dtl[env.name[x]]]) for x in self.type])
        self.max = np.array([len(env.spt[env.dtl[env.name[x]]]) for x in self.type])
