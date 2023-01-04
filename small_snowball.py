# -*- coding: utf-8 -*-
import numpy as np
import pickle as pk
import scipy.optimize as opt


class SmallSnowBall:

    def __init__(self, s0, r, sigma, t, s_kout, premium):
        self.n_days_per_mth = 21
        self.n_days_per_yr = 252
        self.kout_obs_days = np.arange(self.n_days_per_mth, self.n_days_per_yr + 1, self.n_days_per_mth, dtype=int)

        self.s0 = s0  # 期初价，一般为100
        self.r = r  # risk-free annual interest rate
        self.v = sigma  # 波动率
        self.t = t  # 存续期，单位为年，默认为1年
        self.t_days = t * self.n_days_per_yr  # period in day
        self.s_kout = s_kout  # 敲出价
        self.premium = premium  # 期权费，0.01表示名义本金的1%

        self.q = 0.047  # dividend

    def gen_trends_mc(self, rand_fp):
        """从指定的文件中读入并生成MC路径"""
        rand = np.array(pk.load(open(rand_fp, 'rb')))
        # rand = np.random.randn(100000, 252)
        mu = self.r - self.q
        d_t = 1.0 / self.n_days_per_yr
        d_ln_S = (mu - 0.5 * self.v ** 2) * d_t + self.v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = self.s0 * np.exp(ln_S)

        return S_all

    def classify_trends(self, trends):
        # knock out
        kout_bool_matrix = trends[:, self.kout_obs_days] >= self.s_kout
        kout_flags = np.any(kout_bool_matrix, axis=1)
        kout_days_index = kout_bool_matrix[kout_flags, :].argmax(axis=1) if kout_flags.any() else []
        kout_days = self.kout_obs_days[kout_days_index]
        kout_period_in_yr = kout_days / self.n_days_per_yr

        # not knock out
        n_nkout = len(trends) - len(kout_period_in_yr)

        return kout_period_in_yr, n_nkout

    def valuate(self, coupon_rate, kout_period_in_yr, n_paths):
        principal = self.s0

        coupons = principal * coupon_rate * kout_period_in_yr * np.exp(-self.r * kout_period_in_yr)
        pv_kout = np.sum(coupons)

        total_premium = n_paths * principal * self.premium
        return pv_kout - total_premium

    def valuate_due(self, coupon_rate, kout_period_in_yr, n_paths):
        principal = self.s0

        # knock out
        coupons = principal * coupon_rate * kout_period_in_yr * np.exp(-self.r * kout_period_in_yr)
        kout_premium = principal * self.premium * np.exp(-self.r * kout_period_in_yr)
        pv_kout = np.sum(coupons - kout_premium)

        nkout_premium = (n_paths - len(kout_period_in_yr)) * principal * self.premium * np.exp(-self.r * self.t)
        return pv_kout - nkout_premium

    def find_coupon_rate(self, rand_fp):
        trends = self.gen_trends_mc(rand_fp)  # 读入并生成MC路径
        kout_period_in_yr, n_nkout = self.classify_trends(trends)  # 路径分类
        coupon = opt.newton(self.valuate_due, 0.05, args=(kout_period_in_yr, len(trends)))
        print(f"敲出票息率：{coupon:.6%}")
        return coupon
