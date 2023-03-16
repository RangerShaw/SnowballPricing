# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt


class SmallSnowBall:

    def __init__(self, s0, r, q, v, d_r, months, s_kout, fund, up):
        self.days_per_mth = 21
        self.days_per_yr = 252

        self.s0 = s0  # 期初价
        self.r = r  # risk-free annual interest rate
        self.q = q  # dividend
        self.v = v  # 波动率
        self.d_r = d_r
        self.months = months  # 存续期，单位为月
        self.days = months * self.days_per_mth  # period in day
        self.s_kout = s_kout  # 敲出价
        self.premium = fund  # 期权费，0.01表示名义本金的1%
        self.upside = up
        self.kout_obs_days = np.arange(self.days_per_mth, self.days + 1, self.days_per_mth, dtype=int)

    def gen_trends_mc(self, rand_fp, days, s0, r, q, v):
        rand = np.random.randn(100000, days)
        mu = r - q
        d_t = 1.0 / self.days_per_yr
        d_ln_S = (mu - 0.5 * v ** 2) * d_t + v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = s0 * np.exp(ln_S)
        return S_all

    def classify_trends(self, trends, kout_obs_days, s_kout, is_upside):
        # knock out
        kout_bool_matrix = trends[:, kout_obs_days] >= s_kout if is_upside else trends[:, kout_obs_days] <= s_kout
        kout_flags = np.any(kout_bool_matrix, axis=1)
        kout_days_index = kout_bool_matrix[kout_flags, :].argmax(axis=1) if kout_flags.any() else []
        kout_period_map = kout_obs_days / self.days_per_yr
        kout_periods_yr = kout_period_map[kout_days_index]
        return kout_periods_yr

    def valuate(self, coupon_rate, discount_r, funding, kout_periods_yr, kin_period_yr, n_kin):
        principal = 1000
        # knock out
        kout_profit = principal * (coupon_rate - funding) * kout_periods_yr * np.exp(-discount_r * kout_periods_yr)
        pv_kout = np.sum(kout_profit)
        # not knock out
        pv_kin = n_kin * principal * funding * kin_period_yr * np.exp(-discount_r * kin_period_yr)
        return pv_kout - pv_kin

    def find_coupon_rate(self, rand_fp):
        trends = self.gen_trends_mc(rand_fp, self.days, self.s0, self.r, self.q, self.v)
        kout_periods_yr = self.classify_trends(trends, self.kout_obs_days, self.s_kout, self.upside)
        print(f'敲出路径条数: {len(kout_periods_yr)}')
        print(f'平均敲出天数: {np.average(kout_periods_yr) * 365}')
        coupon = opt.newton(self.valuate, 0.05, args=(
            self.d_r, self.premium, kout_periods_yr, self.months / 12, len(trends) - len(kout_periods_yr)))
        print(f"敲出票息率：{coupon:.6%}")
        return coupon


if __name__ == '__main__':
    sb = SmallSnowBall(s0=1834, r=0.012, q=0.0, v=0.15, d_r=0.024, months=12, s_kout=1834 * 1.0, fund=0.0265, up=True)
    sb.find_coupon_rate('')
