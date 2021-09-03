# -*- coding: utf-8 -*-
from snowball import SnowBall


def get_coupon_rate():
    fp = 'data/QuasiRand.pickle'
    sb = SnowBall(100.0, 0.036, 0.3, 1, 75, 100)
    return sb.find_coupon_rate(fp)


def get_beta(coupon_rate):
    pass


if __name__ == '__main__':
    coupon_rate = get_coupon_rate()
    # coupon_rate = 0.23953
