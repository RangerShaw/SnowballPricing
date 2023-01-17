# -*- coding: utf-8 -*-
# from snowball import SnowBall
from small_snowball import *
from digital import Digital
import xlwings as xw
import pandas as pd
import numpy as np

trade_days = pd.read_csv('trade_days.csv', encoding='utf-8', parse_dates=['trading_days'])
trade_days.head()
trade_days.dtypes

wb = xw.Book("myproject.xlsm")
sheet = xw.Book("myproject.xlsm").sheets[0]
date_paras = sheet.range('B12').options(transpose=True, expand='table').value
[obs_dates, _] = date_paras

market_paras = sheet['C3:C7'].value
product_paras = sheet['F3:F8'].value
[today, S0, v, r, q] = market_paras
[s_date, e_date, funding, coupon_rate, s_kout, principle] = product_paras
print(market_paras)
print(product_paras)
# sb = SmallSnowBall(5952.45, 0.0285, 0.17, 1, 5952.45, 0.025)
# pv = sb.mark_to_market(S0, s_kout, funding, coupon_rate, r, q, v, 20)
# sheet['G2'].value = pv

sb365 = SmallsbM2M()
pv1 = sb365.m2m_365(principle, S0, s_kout, funding, coupon_rate, r, q, v, today, np.array(obs_dates), s_date, e_date)
print(pv1)

fp = 'data/QuasiRand.pickle'


# def get_sb_coupon_rate():
#     fp = 'data/QuasiRand.pickle'
#     sb = SnowBall(100.0, 0.036, 0.3, 1, 75, 100)
#     return sb.find_coupon_rate(fp)


def get_smallsb_coupon_rate():
    sb = SmallSnowBall(5952.45, 0.0285, 0.17, 1, 5952.45, 0.025)
    return sb.find_coupon_rate(fp)


def get_digital_coupon_rate():
    sb = Digital(5952.45, 0.0235, 0.17, 1, 5952.45, 0.01)
    return sb.find_coupon_rate(fp)


def get_smallsb_M2M():
    sb = SmallSnowBall(5952.45, 0.0285, 0.17, 1, 5952.45, 0.025)
    return sb.mark_to_market(10000, 5952.45, 0.025, 0.057, 0.0235, 0.05227, 0.17, 251)


if __name__ == '__main__':
    # coupon_rate = get_sb_coupon_rate()
    # smallsnowball_coupon_rate = get_smallsb_coupon_rate()
    # digital_coupon_rate = get_digital_coupon_rate()
    # get_smallsb_M2M()
    pass