# -*- coding: utf-8 -*-
import numpy as np
import pickle as pk
import scipy.optimize as opt


class Digital:

    def __init__(self, s0, r, sigma, t, K, premium):
        self.n_days_per_mth = 21
        self.n_days_per_yr = 252
        self.kout_obs_days = np.arange(self.n_days_per_mth, self.n_days_per_yr + 1, self.n_days_per_mth, dtype=int)
        self.paths = None

        self.s0 = s0  # 期初价，一般为100
        self.r = r  # risk-free annual interest rate
        self.v = sigma  # 波动率
        self.t = t  # 存续期，单位为年，默认为1年
        self.t_days = t * self.n_days_per_yr  # period in day
        self.K = K  # 行权价
        self.premium = premium  # 期权费，0.01表示名义本金的1%
        self.q = 0.03644  # dividend

    def gen_trends_mc(self, rand_fp):
        """从指定的文件中读入并生成MC路径"""
        # rand = np.array(pk.load(open(rand_fp, 'rb')))
        rand = np.random.randn(100000, 252)
        mu = self.r - self.q
        d_t = 1.0 / self.n_days_per_yr
        d_ln_S = (mu - 0.5 * self.v ** 2) * d_t + self.v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = self.s0 * np.exp(ln_S)

        return S_all

    def classify_trends(self, trends):
        # knock out
        kout_bool_matrix = trends[:, -1] >= self.K
        n_kout = np.sum(kout_bool_matrix)
        n_nkout = np.sum(trends[:, -1] < self.K)

        return n_kout, n_nkout

    def valuate(self, coupon_rate, kout_period_in_yr, n_paths):
        principal = self.s0

        coupons = principal * coupon_rate * kout_period_in_yr * np.exp(-self.r * kout_period_in_yr)
        pv_kout = np.sum(coupons)

        total_premium = n_paths * principal * self.premium
        return pv_kout - total_premium

    def valuate_due(self, coupon_rate, n_kout, n_nkout):
        principal = self.s0

        # knock out
        coupons = n_kout * principal * coupon_rate * self.t * np.exp(-self.r * self.t)
        premium = (n_kout + n_nkout) * principal * self.premium * self.t * np.exp(-self.r * self.t)
        return coupons - premium

    def find_coupon_rate(self, rand_fp):
        self.paths = self.gen_trends_mc(rand_fp)  # 读入并生成MC路径
        n_kout, n_nkout = self.classify_trends(self.paths)  # 路径分类
        print(f"总路径数：{len(self.paths)}，获得收益路径数：{n_kout}，占比{n_kout / len(self.paths):.3}")
        # print(f"未获得收益路径数：{n_nkout}")
        coupon = opt.newton(self.valuate_due, 0.05, args=(n_kout, n_nkout))
        # print(f"敲出票息率：{coupon:.6%}")
        return coupon

