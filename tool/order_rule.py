import numpy as np

"""
FIFO 待加工工件越早达到越优先
SPT 待加工工件加工时间越短越优先
LWR 待加工工件剩余加工时间越短越优先
EDD 待加工工件交货期最早的工件优先
SRPT 待加工工件剩余处理时间越短越优先
HRN 待加工工件等待时间和加工时间的差与加工时间
之比越大的工件越优先
MDD 待加工工件修正交货期越小的工件越优先
CR 待加工工件工序临界比越小的工件越优先
"""


def fifo(order_select):
    return order_select.iloc[np.argmin(order_select["arrive"])]


def spt(order_select):
    return order_select.iloc[np.argmin(order_select["process"])]


def lwr(order_select):
    return order_select.iloc[np.argmin(order_select["remain"])]


def edd(order_select):
    return order_select.iloc[np.argmin(order_select["expected"])]


def srpt(order_select):
    return order_select.iloc[np.argmin(order_select["remain"] / order_select["total"])]
