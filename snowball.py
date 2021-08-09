# -*- coding: utf-8 -*-
import math

import numpy as np
import pickle as pk
import scipy.optimize as opt


class SnowBall:

    def __init__(self, s0, r, sigma, t, in_price, out_price):
        self.n_days_per_month = 21
        self.n_days_per_year = 252
        self.knock_out_obs_days = np.arange(21, 253, self.n_days_per_month, dtype=int)

        self.s0 = s0
        self.r = r  # riskless annual interest rate
        self.v = sigma
        self.t = t  # duration in year
        self.t_days = t * self.n_days_per_year  # duration in days
        self.p_knock_in = in_price
        self.p_knock_out = out_price

        self.margin_rate = 1.0
        self.q = 0.0  # dividend

    def gen_trends_mc(self, rand):
        mu = self.r - self.q
        d_t = 1.0 / self.n_days_per_year
        d_ln_S = (mu - 0.5 * self.v ** 2) * d_t + self.v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = self.s0 * np.exp(ln_S)

        return S_all

    def classify_trends(self, trends):
        # knock out
        kout_bool_matrix = trends[:, self.knock_out_obs_days] >= self.p_knock_out
        kout_flags = np.any(kout_bool_matrix, axis=1)
        kout_days_index = kout_bool_matrix[kout_flags, :].argmax(axis=1) if kout_flags.any() else []
        kout_days = self.knock_out_obs_days[kout_days_index]

        # knock in not out: break even or loss
        kin_bool_matrix = trends <= self.p_knock_in
        kin_flags = np.any(kin_bool_matrix, axis=1)
        kin_nkout_flags = kin_flags & ~kout_flags
        kin_nkout_loss_flags = (trends[:, self.t_days] < self.s0) & kin_nkout_flags
        n_kin_nkout_even = kin_nkout_flags.sum() - kin_nkout_loss_flags.sum()
        kin_nkout_loss_end_prices = trends[kin_nkout_loss_flags, self.t_days]
        kin_nkout_loss_rate = kin_nkout_loss_end_prices / self.s0 - 1

        # not knock in or out
        nkin_nkout_flags = ~kin_flags & ~kout_flags
        n_nkin_nkout = nkin_nkout_flags.sum()

        return kout_days, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout

    def valuate(self, coupon_rate, kout_days, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout):
        # knock out
        kout_duration_ration = kout_days / self.n_days_per_year
        pv_kout = (self.s0 * coupon_rate * kout_duration_ration * np.exp(-self.r * kout_duration_ration)).sum()

        # # knock in: break even
        # pv_kin_even = self.s0 * (-1 + np.exp(-self.r * self.t)) * n_kin_nkout_even

        # knock in: loss
        pv_kin_loss = (self.s0 * kin_nkout_loss_rate * np.exp(-self.r * self.t)).sum()

        # not knock in or out
        pv_nkin_nkout = self.s0 * coupon_rate * self.t * np.exp(-self.r * self.t) * n_nkin_nkout

        return pv_kout + pv_kin_loss + pv_nkin_nkout

    def solve_mc(self, rand_fp):
        rand = np.array(pk.load(open(rand_fp, 'rb')))
        trends = self.gen_trends_mc(rand)

        kout_days, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout = self.classify_trends(trends)
        coupon = opt.newton(self.valuate, 0.2, args=(kout_days, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout))

        print(coupon)
        print(self.valuate(coupon, kout_days, n_kin_nkout_even, kin_nkout_loss_rate, n_nkin_nkout))
