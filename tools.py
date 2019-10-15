import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm
import itertools
from multiprocessing.pool import ThreadPool
import time
import threading


def pc_combine(listA, listB):
    list_com = []
    for i in listA:
        for j in listB:
            list_com.append((i,j))
    return list_com

def parallel(lock, position, total):
    text = "progresser #{}".format(position)
    with lock:
        progress = tqdm(
            total=total,
            position=position,
            desc=text,
        )
    for _ in range(0, total, 5):
        with lock:
            progress.update(5)
        time.sleep(0.1)
    with lock:
        progress.close()


def demo():
    pool = ThreadPool(6)
    tasks = range(6)
    lock = threading.Lock()
    for i, url in enumerate(tasks, 1):
        pool.apply_async(demo, args=(lock, i, 100))
    pool.close()
    pool.join()
