# -*- coding: utf-8 -*-
# from snowball import SnowBall
from small_snowball import SmallSnowBall
from digital import Digital

# def get_sb_coupon_rate():
#     fp = 'data/QuasiRand.pickle'
#     sb = SnowBall(100.0, 0.036, 0.3, 1, 75, 100)
#     return sb.find_coupon_rate(fp)


def get_smallsb_coupon_rate():
    fp = 'data/QuasiRand.pickle'
    sb = SmallSnowBall(5952.45, 0.0285, 0.17, 1, 5952.45, 0.025)
    return sb.find_coupon_rate(fp)

def get_digital_coupon_rate():
    fp = 'data/QuasiRand.pickle'
    sb = Digital(5952.45, 0.0235, 0.17, 1, 5952.45, 0.01)
    return sb.find_coupon_rate(fp)


if __name__ == '__main__':
    # coupon_rate = get_sb_coupon_rate()
    smallsnowball_coupon_rate = get_smallsb_coupon_rate()
    # digital_coupon_rate = get_digital_coupon_rate()
