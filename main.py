# -*- coding: utf-8 -*-
# from snowball import SnowBall
from small_snowball import SmallSnowBall


# def get_sb_coupon_rate():
#     fp = 'data/QuasiRand.pickle'
#     sb = SnowBall(100.0, 0.036, 0.3, 1, 75, 100)
#     return sb.find_coupon_rate(fp)


def get_smallsb_coupon_rate():
    fp = 'data/QuasiRand.pickle'
    sb = SmallSnowBall(100.0, 0.036, 0.17, 1, 100, 0.005)
    return sb.find_coupon_rate(fp)


if __name__ == '__main__':
    # coupon_rate = get_sb_coupon_rate()
    smallsnowball_coupon_rate = get_smallsb_coupon_rate()
