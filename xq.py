# solve coupon
# calculate pv

import numpy as np
import time
import scipy.optimize as opt


class xq:
    def __init__(self, ki_b, ko_b, lock, ko_dec):
        # stocks
        self.S0 = 100
        self.r = 0.032
        self.q = 0
        self.mu = self.r - self.q
        self.sig = 0.135
        self.T = 2
        self.ko_dec = ko_dec

        # snowball
        self.np = 1  # nominal principle
        self.mratio = 1
        self.margin = self.np * self.mratio
        self.no_shares = self.np / self.S0  # number of shares

        self.obs_array = np.array(
            [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365, 396, 424, 455, 485, 516, 546, 577, 608, 638, 669,
             699, 730])

        self.ko_obs = self.obs_array[lock:]

        self.ki_b = ki_b

        self.dec_array = np.arange(len(self.ko_obs)) * ko_dec

        # self.ko_array=np.arange(ko_b,0,-ko_dec)
        # self.ko_b=self.ko_array[:len(self.ko_obs)]
        self.ko_b = ko_b - self.dec_array

        # monte carlo
        # take 1 interval == 1 day to make ko_obs easier
        self.ndays1year = 365
        self.no_intervals = int(self.T * self.ndays1year)
        self.dt = 1 / self.ndays1year
        self.no_mc = 10 ** 5

        self.df_vec = np.exp(-self.r * self.dt * np.arange(self.no_intervals + 1))

        self.path_pt = self.GBM_pt()

    def GBM_pt(self):
        """
        Generate Geometric Brownian Motion Paths
        """
        np.random.seed(0)
        xi = np.random.normal(size=(self.no_mc, self.no_intervals))
        # xi = self.QuasiRandSeed(self.no_mc, self.no_intervals)
        R = np.exp((self.mu - self.sig ** 2 / 2) * self.dt + self.sig * np.sqrt(self.dt) * xi)
        path = np.ones((self.no_mc, self.no_intervals + 1), dtype=float)
        path[:, 1:] = np.cumprod(R, axis=1)
        return path

    def pv(self, coupon):
        path = self.S0 * self.path_pt

        ko_mat = path[:, self.ko_obs] >= self.ko_b
        ki_mat = path <= self.ki_b
        # case ko
        ko = ko_mat.any(axis=1)
        ki = ki_mat.any(axis=1)

        if ko.any():
            ko_idx = ko_mat[ko, :].argmax(axis=1)
            ko_day = self.ko_obs[ko_idx]
        else:
            ko_day = np.array([], dtype=int)

        nko = ~ko
        # case ki_nko
        nko_ki = nko & ki
        # case nki_nko
        nko_nki = nko & (~ki)

        p_ko = np.sum(-self.margin + (self.np * coupon * ko_day * self.dt + self.margin) * self.df_vec[ko_day])

        p_ki_nko = np.sum(
            -self.margin + (self.margin + np.minimum((path[nko_ki, -1] - self.S0) * self.no_shares, 0)) * self.df_vec[
                self.no_intervals])

        p_nki_nko = (-self.margin + (self.np * coupon * self.no_intervals * self.dt + self.margin) * self.df_vec[
            self.no_intervals]) * (nko_nki.sum())

        # print(p1,p2,p3)
        pv = (p_ko + p_ki_nko + p_nki_nko) / self.no_mc
        return pv

    def coupon_solver(self):
        coupon = opt.newton(self.pv, 0.5, args=())
        return coupon
