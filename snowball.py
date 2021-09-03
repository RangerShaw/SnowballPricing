# -*- coding: utf-8 -*-
import numpy as np
import pickle as pk
import scipy.optimize as opt


class SnowBall:

    def __init__(self, s0, r, sigma, t, s_kin, s_kout):
        self.n_days_per_mth = 21
        self.n_days_per_yr = 252
        self.kout_obs_days = np.arange(self.n_days_per_mth, self.n_days_per_yr + 1, self.n_days_per_mth, dtype=int)

        self.s0 = s0
        self.r = r  # risk-free annual interest rate
        self.v = sigma
        self.t = t  # period in year
        self.t_days = t * self.n_days_per_yr  # period in day
        self.s_kin = s_kin
        self.s_kout = s_kout

        self.margin_rate = 1.0
        self.q = 0.0  # dividend

    def gen_trends_mc(self, rand):
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

        # knock in not out: break even or loss
        kin_bool_matrix = trends <= self.s_kin
        kin_flags = np.any(kin_bool_matrix, axis=1)
        kin_nkout_flags = kin_flags & ~kout_flags
        kin_nkout_loss_flags = (trends[:, self.t_days] < self.s0) & kin_nkout_flags
        n_kin_nkout_even = kin_nkout_flags.sum() - kin_nkout_loss_flags.sum()
        kin_nkout_loss_end_prices = trends[kin_nkout_loss_flags, self.t_days]
        kin_nkout_loss_rate = kin_nkout_loss_end_prices / self.s0 - 1

        # not knock in or out
        nkin_nkout_flags = ~kin_flags & ~kout_flags
        n_nkin_nkout = nkin_nkout_flags.sum()

        return kout_period_in_yr, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout

    def valuate(self, coupon_rate, kout_period_in_yr, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout):
        # knock out
        pv_kout = np.sum((self.s0 * coupon_rate * kout_period_in_yr * np.exp(-self.r * kout_period_in_yr)))

        # # knock in: break even
        # pv_kin_even = self.s0 * (-1 + np.exp(-self.r * self.t)) * n_kin_nkout_even

        # knock in: loss
        pv_kin_loss = np.sum((self.s0 * kin_nkout_loss_rate * np.exp(-self.r * self.t)))

        # not knock in or out
        pv_nkin_nkout = self.s0 * coupon_rate * self.t * np.exp(-self.r * self.t) * n_nkin_nkout

        return pv_kout + pv_kin_loss + pv_nkin_nkout

    def find_coupon_rate(self, rand_fp):
        rand = np.array(pk.load(open(rand_fp, 'rb')))
        trends = self.gen_trends_mc(rand)

        kout_period_in_yr, n_kin_nkout_even, kin_nkout_loss_yield, n_nkin_nkout = self.classify_trends(trends)
        coupon = opt.newton(self.valuate, 0.2,
                            args=(kout_period_in_yr, n_kin_nkout_even, kin_nkout_loss_yield, n_nkin_nkout))

        print(coupon)
        print(self.valuate(coupon, kout_period_in_yr, n_kin_nkout_even, kin_nkout_loss_yield, n_nkin_nkout))
        return coupon

    def gen_price_matrix(self, rand_fp):
        price_matrix = np.zeros((101, 253))

        rand = np.array(pk.load(open(rand_fp, 'rb')))
        trends = self.gen_trends_mc(rand)

        for i in range(0, 253):
            trends_t = rand[:, :253 - i]

        return price_matrix
