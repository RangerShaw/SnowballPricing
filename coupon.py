# -*- coding: utf-8 -*-
import numpy as np
import time
from xq import *


def coupon_solver(eps=0.0001):
    args = [[75, 100, 0, 0]]
    # args.append([75,100,0,0.5])
    # args.append([75,103,0,0.5])
    # args.append([75,100,2,0])
    # args.append([75,100,2,0.5])
    # args.append([75,103,2,0.5])

    coupon = []

    for i in range(len(args)):
        (ki_b, ko_b, lock, ko_dec) = args[i]
        product = xq(ki_b, ko_b, lock, ko_dec)
        coupon.append(product.coupon_solver())
    return coupon


if __name__ == '__main__':
    t1 = time.time()
    coupon = coupon_solver()
    t2 = time.time()

    print(coupon)
    print(t2 - t1)
