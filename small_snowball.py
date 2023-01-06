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

        self.q = 0.05227  # dividend

    def gen_trends_mc(self, rand_fp, days, S0, r, q, v):
        """从指定的文件中读入并生成MC路径"""
        rand = np.array(pk.load(open(rand_fp, 'rb'))) if rand_fp != '' and days == 252 else np.random.randn(100000,
                                                                                                            days)
        mu = r - q
        d_t = 1.0 / self.n_days_per_yr
        d_ln_S = (mu - 0.5 * v ** 2) * d_t + v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = S0 * np.exp(ln_S)

        return S_all

    def classify_trends(self, trends, kout_obs_days, s_kout, total_days, day_offset):
        # knock out
        kout_bool_matrix = trends[:, kout_obs_days] >= s_kout
        kout_flags = np.any(kout_bool_matrix, axis=1)
        kout_days_index = kout_bool_matrix[kout_flags, :].argmax(axis=1) if kout_flags.any() else []
        kout_days = kout_obs_days[kout_days_index]
        kout_period_in_yr = (kout_days + day_offset) / total_days

        return kout_period_in_yr

    def valuate_old(self, coupon_rate, kout_period_in_yr, n_paths):
        principal = self.s0

        coupons = principal * coupon_rate * kout_period_in_yr * np.exp(-self.r * kout_period_in_yr)
        pv_kout = np.sum(coupons)

        total_premium = n_paths * principal * self.premium
        return pv_kout - total_premium

    def valuate(self, coupon_rate, r, funding, kout_period_in_yr, kout_discount_yr, discount_year, n_paths):
        principal = 100

        # knock out
        coupons = principal * coupon_rate * kout_period_in_yr * np.exp(-r * kout_discount_yr)
        kout_premium = principal * funding * kout_period_in_yr * np.exp(-r * kout_discount_yr)
        pv_kout = np.sum(coupons - kout_premium)

        # not knock out
        nkout_premium = (n_paths - len(kout_period_in_yr)) * principal * funding * np.exp(-r * discount_year)
        return pv_kout - nkout_premium

    def find_coupon_rate(self, rand_fp):
        trends = self.gen_trends_mc(rand_fp, self.t * self.n_days_per_yr, self.s0, self.r, self.q, self.v)
        kout_period_yr = self.classify_trends(trends, self.kout_obs_days, self.s_kout, self.t_days, 0)
        coupon = opt.newton(self.valuate, 0.05,
                            args=(self.r, self.premium, kout_period_yr, kout_period_yr, self.t, len(trends)))
        print(f"敲出票息率：{coupon:.6%}")
        return coupon

    def mark_to_market(self, S1, s_kout_, funding_, coupon_rate_, r_, q_, v_, days):
        if days in self.kout_obs_days and S1 >= s_kout_:
            pv = (coupon_rate_ - funding_) * days / self.t_days
            print(f'M2M为名义本金的：{pv:.6%}')
            return pv

        kout_obs_days_ = self.kout_obs_days - days
        kout_obs_days_ = kout_obs_days_[kout_obs_days_ > 0]
        left_days = self.t * self.n_days_per_yr - days

        paths = self.gen_trends_mc('', left_days, S1, r_, q_, v_)
        kout_period_yr = self.classify_trends(paths, kout_obs_days_, s_kout_, self.t_days, days)
        kout_discount_yr = kout_period_yr - (days / self.n_days_per_yr)

        pv = self.valuate(coupon_rate_, r_, funding_, kout_period_yr, kout_discount_yr, left_days / self.n_days_per_yr,
                          len(paths))
        pv_ratio = pv / 100 / len(paths)
        print(f'M2M为名义本金的：{pv_ratio:.6%}')
        return pv_ratio


class SmallsbM2M:

    def __init__(self):
        pass

    def gen_trends_mc(self, days, S0, r, q, v):
        rand = np.random.randn(100000, days)
        mu = r - q
        d_t = 1.0 / 252
        d_ln_S = (mu - 0.5 * v ** 2) * d_t + v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = S0 * np.exp(ln_S)

        return S_all

    def m2m_365(self, principle, s0, s_kout, funding, coupon_rate, r, q, v, today, obs_days, ops_day):
        if 0 in obs_days and s0 >= s_kout:
            pv = (coupon_rate - funding) * principle * ops_day / 365
            print(f'M2M价值：{pv:.6%}')
            return pv

        obs_days = obs_days[obs_days > 0]
        paths = self.gen_trends_mc(obs_days[-1], s0, r, q, v)
