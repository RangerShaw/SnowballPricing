# -*- coding: utf-8 -*-
import math

import numpy as np
import pickle as pk
from tqdm import tqdm


class Trend:
    strike_out = False
    strike_in = False
    duration_ratio = 1.0
    ini_price = 1.0
    end_price = 1.0
    return_rate = 1.0

    def __init__(self, sout, sin, return_rate):
        self.strike_out = sout
        self.strike_in = sin
        self.return_rate = return_rate


class SnowBall:

    def __init__(self, s0, r, sigma, t, in_price, out_price):
        self.s0 = s0
        self.r = r  # riskless annual interest rate
        self.v = sigma
        self.t = t  # duration in year
        self.p_knock_in = in_price
        self.p_knock_out = out_price
        self.q = 0.0  # dividend
        self.n_days_per_month = 21
        self.n_days_per_year = 252
        self.knock_out_obs_days = np.arange(21, 253, self.n_days_per_month, dtype=int)
        self.margin_rate = 1.0

    def gen_trends_mc(self, rand):
        mu = self.r - self.q
        dS = (mu - 0.5 * self.v ** 2) * 1.0 / self.n_days_per_year + self.v * np.sqrt(1.0 / self.n_days_per_year) * rand

        dS = np.insert(dS, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        Sr = np.cumsum(dS, axis=1)
        S_all = self.s0 * np.exp(Sr)

        return S_all

    def classify_trends(self, trends):
        knock_out_bools = trends[:, self.knock_out_obs_days] >= self.p_knock_out
        knock_in_bools = trends <= self.p_knock_in

        knock_out_bool = np.any(knock_out_bools, axis=1)
        knock_in_bool = np.any(knock_in_bools, axis=1)


        strike_out_duration_ratio = np.array([], dtype=float)
        strike_in_loss_trends = np.array([], dtype=float)
        n_strike_in_break_even_trends = 0
        n_stable_trends = 0

        for j in tqdm(range(0, len(trends))):
            trend = trends[j]
            strike_in = False

            for j in range(1, len(trend)):
                if (j % self.n_days_per_month) % self.knock_out_obs_days == 0 and trend[j] >= self.p_knock_out:
                    strike_out_duration_ratio = np.append(strike_out_duration_ratio, [j / self.n_days_per_year])
                    continue
                if trend[j] <= self.p_knock_in:
                    strike_in = True

            if not strike_in:
                n_stable_trends += 1
            elif trend[-1] >= trend[0]:
                n_strike_in_break_even_trends += 1
            else:
                strike_in_loss_trends = np.append(strike_in_loss_trends, [trend[-1] / trend[0] - 1])

        return strike_out_duration_ratio, strike_in_loss_trends, n_stable_trends

    def valuate(self, n_trend, coupon_rate, strike_in_trends, strike_out_loss_trends, n_stable_trends):
        loss = strike_out_loss_trends.sum()
        value = n_stable_trends * coupon_rate + strike_out_loss_trends.sum() + strike_in_trends.sum() * coupon_rate
        value *= self.s0 * np.exp(-self.r * self.t)
        return value / n_trend

    def solve_mc(self, rand_fp):
        rand = np.array(pk.load(open(rand_fp, 'rb')))
        rand = rand[:10000]
        trends = self.gen_trends_mc(rand)
        strike_in_trends, strike_out_loss_trends, n_stable_trends = self.classify_trends(trends)
        value = self.valuate(10000, 0.1, strike_in_trends, strike_out_loss_trends, n_stable_trends)
        print(value)
