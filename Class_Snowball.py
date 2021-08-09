# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:09:14 2021

@author: Monica_C_HE
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from matplotlib import pylab, cm
import matplotlib.pyplot as plt
import time


class Snowball:
    def __init__(self, _r, _q, _sigma, _t, KO, KI, margin=1, _Ndays1year=252, _timetype='years'):
        '''
        -------------------------------------------------
        _typeflag：期权类型，提供以下2类：
            1) 'c'=call,    2) 'p'=put
        _timetpye：输入时间参数的单位，提供两个参数：years与days
            years代表单位为年，days代表单位为交易日。每年252个交易日。

        '''
        self.r = _r
        self.q = _q
        self.v = _sigma
        self.timetype = _timetype
        self.Ndays1year = _Ndays1year
        self.Ndays1mon = int(_Ndays1year / 12)
        self.KO = KO
        self.KI = KI
        self.margin = margin

        if self.timetype == 'days':
            self.t = _t / self.Ndays1year
        elif self.timetype == 'years':
            self.t = _t
        else:
            raise (Exception('_timetpye目前仅提供两个参数可选：years与days'))

    def QuasiRandSeed(self, filename, MC_lens, T_lens):
        '''
        ---------------------------------------------------------
        此函数用于使用外部文件中定义的随机数种子
        
        '''
        QuasiRand = np.array(pd.read_pickle(filename))
        if MC_lens > len(QuasiRand):
            print(" MC length is too long!")
        RandSeed = QuasiRand[:MC_lens, :T_lens]
        return RandSeed

    def MonteCarloGenerate(self, St, filename, MC_lens, T_lens, MCMethod="Sobol"):
        '''
        ---------------------------------------------------------
        此函数用于使用MC方法生成模拟序列
        MC方法可以选择"Sobol"或其他，使用Sobol方法需要给出对应的种子文件地址
        若使用普通方法，filename和MCMethod参数可以随意输入
        
        '''
        self.s = St

        if MCMethod == "Sobol":
            Rand = self.QuasiRandSeed(filename, MC_lens, T_lens)
        else:
            Rand = np.random.randn(MC_lens, T_lens)

        mu = self.r - self.q
        dS = (mu - 0.5 * self.v ** 2) * 1.0 / self.Ndays1year + self.v * np.sqrt(1.0 / self.Ndays1year) * Rand

        dS = np.insert(dS, 0, values=np.zeros(MC_lens), axis=1)

        Sr = np.cumsum(dS, axis=1)

        SAll = St * np.exp(Sr)

        return SAll

    def MCSolver(self, SAll, obs_day=21):
        '''
        ---------------------------------------------------------
        此函数用于使用MC方法计算雪球估值
        SAll：已有的模拟序列
        obs_day：每月第k天观察
        frozendays：锁定天数

        '''

        # [SM, SN] = SAll.shape
        # self.obs_day = obs_day
        # KO_list = list(range(SN))[self.obs_day::self.Ndays1mon]

        # S_obs_daily = SAll[:, 0:]
        # S_obs_monthly = SAll[:, KO_list]

        # terminate = np.ones(SM) * self.t

        # is_KO = (S_obs_monthly >= self.KO * self.s) * 1
        # temp = np.where(is_KO==1)  # 0: row; 1: col
        # arg_KO = np.unique(temp[0], return_index=True)[1]
        # s_KO = temp[0][arg_KO]
        # t_KO = temp[1][arg_KO]
        # terminate[s_KO] = (t_KO + 1) / 12

        # is_KI = ((S_obs_daily <= self.KI * self.s) * 1).max(axis=1)
        # s_KI = np.where(is_KI==1)[0]

        # idx_ = list(set(s_KI).difference(set(s_KO)))

        def fun(coupon):
            # Pnf = terminate * coupon
            # Pnf[idx_] = np.minimum(SAll[idx_, -1] / self.s - 1, 0)
            # # Pnf -= terminate * self.margin * self.r
            # Pnf -= self.margin * (np.exp(self.r * terminate) - 1)
            # OutPut = np.mean(Pnf * np.exp(-self.r * terminate))
            OutPut = self.Pnf(SAll, coupon, HAVE_KI=False)
            return OutPut ** 2

        return minimize(fun, 0.2, method='TNC').x

    def Pnf(self, _SAll, coupon, HAVE_KI=False, **kwargs):

        s = self.s
        try:
            t = kwargs['t']
        except:
            t = self.t

        SAll = _SAll[:, :int(t * self.Ndays1year) + 1]
        [SM, SN] = SAll.shape
        start = (SN - 1) % self.Ndays1mon
        KO_list = list(range(SN))[start::self.Ndays1mon]
        if SN > self.Ndays1year * self.t: KO_list = KO_list[1:]

        S_obs_daily = SAll
        S_obs_monthly = SAll[:, KO_list]

        discount = np.ones(SM) * t

        is_KO = (S_obs_monthly >= self.KO * s) * 1
        temp = np.where(is_KO == 1)  # 0: row; 1: col
        arg_KO = np.unique(temp[0], return_index=True)[1]
        s_KO = temp[0][arg_KO]
        t_KO = temp[1][arg_KO]

        passed = self.t * 12 - len(KO_list)
        terminate = np.ones(SM) * self.t
        terminate[s_KO] = (t_KO + 1 + passed) / 12

        discount[s_KO] = np.array(KO_list)[t_KO.tolist()] / self.Ndays1year

        is_KI = ((S_obs_daily <= self.KI * s) * 1).max(axis=1)
        s_KI = np.where(is_KI == 1)[0]

        if HAVE_KI:
            pnf = np.minimum(SAll[:, -1] / s - 1, 0)
            pnf[s_KO] = terminate[s_KO] * coupon
        elif not HAVE_KI:
            idx_ = list(set(s_KI).difference(set(s_KO)))
            pnf = terminate * coupon
            pnf[idx_] = np.minimum(SAll[idx_, -1] / s - 1, 0)

        # pnf -= terminate * self.margin * self.r
        pnf -= self.margin * (np.exp(self.r * terminate) - 1)
        OutPut = np.mean(pnf * np.exp(-self.r * discount))
        # OutPut = pnf * np.exp(-self.r * discount)

        return OutPut

    def Greeks(self, coupon, Srange, Trange, HAVE_KI=False, MC_lens=100000, SAll=None):
        def fun(args):
            t, s = args
            SAllnew = SAll * s / SAll[0, 0]
            return self.Pnf(SAllnew, coupon, HAVE_KI, t=t)

        grid = [(i, j) for i in Trange for j in Srange]

        pnf = np.array(list(map(fun, grid))).reshape(-1, len(Srange))

        delta_m = np.diff(pnf, axis=1)
        gamma_m = np.diff(np.diff(pnf, axis=1), axis=1)
        return pnf, delta_m, gamma_m

    def DeltaHedge(self, SAll, DeltaNoKI, Coupon, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: vega = Vanilla.vega(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同


        '''
        [SM, SN] = SAll.shape
        # SRange = SAll.reshape(-1,1)
        # DeltaNoKI.loc[0] = np.zeros(DeltaNoKI.shape[1])
        DeltaNoKI.sort_index(ascending=False, inplace=True)

        if not kwargs['isKI'] and not kwargs['isKO']:
            paths = []
            for i in range(SM):
                kox = SN - 1
                delta = np.diag(DeltaNoKI.loc[:, SAll[i]].values)
                cf = - Coupon * self.t + self.margin * (np.exp(self.r * kox / self.Ndays1year) - 1)
                cf -= delta[0] * SAll[i, 0] * np.exp(self.r * DeltaNoKI.index[0])
                cf -= (np.diff(delta) * SAll[i, 1:kox + 1] * np.exp(
                    self.r * np.array(DeltaNoKI.index[1:kox + 1]))).sum()
                cf += delta[-1] * SAll[i, kox]

                paths.append(cf)
            return np.array([SAll[:, -1].tolist(), paths])

        if not kwargs['isKI'] and kwargs['isKO']:
            paths = []
            for i in range(SM):
                KO_list = list(range(SN))[self.obs_day::self.Ndays1mon]
                kox = (((SAll[i, KO_list] >= self.KO * self.s).cumsum() == 0).sum() + 1) * self.Ndays1mon
                delta = np.diag(DeltaNoKI.loc[:, SAll[i, :kox + 1]].values)

                cf = - Coupon * kox / self.Ndays1year + self.margin * (np.exp(self.r * kox / self.Ndays1year) - 1)
                cf -= delta[0] * SAll[i, 0] * np.exp(self.r * DeltaNoKI.index[0])
                cf -= (np.diff(delta) * SAll[i, 1:kox + 1] * np.exp(
                    self.r * np.array(DeltaNoKI.index[1:kox + 1]))).sum()
                cf += delta[-1] * SAll[i, kox]

                paths.append(cf)
            return np.array([SAll[:, -1].tolist(), paths])

        DeltaHaveKI = kwargs['DeltaHaveKI']
        # DeltaHaveKI.loc[0] = np.zeros(DeltaHaveKI.shape[1])
        DeltaHaveKI.sort_index(ascending=False, inplace=True)

        if kwargs['isKI'] and not kwargs['isKO']:
            paths = []
            for i in range(SM):
                kix = int(np.argwhere(SAll[i] <= self.KI * self.s)[0])
                kox = SN - 1

                noki = SAll[i, :kix]
                haveki = SAll[i, kix:kox + 1]
                delta0 = np.diag(DeltaNoKI.loc[:, noki].values)
                delta1 = np.diag(DeltaHaveKI.loc[:, haveki].iloc[kix:].values)
                delta = np.concatenate((delta0, delta1))

                cf = np.max([0, - SAll[i, -1] / SAll[i, 0] + 1]) + self.margin * (
                            np.exp(self.r * kox / self.Ndays1year) - 1)
                cf -= delta[0] * SAll[i, 0] * np.exp(self.r * DeltaNoKI.index[0])
                cf -= (np.diff(delta) * SAll[i, 1:kox + 1] * np.exp(
                    self.r * np.array(DeltaNoKI.index[1:kox + 1]))).sum()
                cf += delta[-1] * SAll[i, kox]

                paths.append(cf)
            return np.array([SAll[:, -1].tolist(), paths])

        if kwargs['isKI'] and kwargs['isKO']:
            paths = []
            for i in range(SM):
                kix = int(np.argwhere(SAll[i] <= self.KI * self.s)[0])
                KO_list = list(range(SN))[self.obs_day::self.Ndays1mon]
                kox = (((SAll[i, KO_list] >= self.KO * self.s).cumsum() == 0).sum() + 1) * self.Ndays1mon

                noki = SAll[i, :kix]
                haveki = SAll[i, kix:kox + 1]
                delta0 = np.diag(DeltaNoKI.loc[:, noki].values)
                delta1 = np.diag(DeltaHaveKI.loc[:, haveki].iloc[kix:].values)
                delta = np.concatenate((delta0, delta1))

                cf = - Coupon * kox / self.Ndays1year + self.margin * (np.exp(self.r * kox / self.Ndays1year) - 1)
                cf -= delta[0] * SAll[i, 0] * np.exp(self.r * DeltaNoKI.index[0])
                cf -= (np.diff(delta) * SAll[i, 1:kox + 1] * np.exp(
                    self.r * np.array(DeltaNoKI.index[1:kox + 1]))).sum()
                cf += delta[-1] * SAll[i, kox]

                paths.append(cf)
            return np.array([SAll[:, -1].tolist(), paths])


def Plot3D(df, cmap=cm.PiYG):
    x = df.index.tolist()
    y = df.columns.tolist()
    x, y = np.meshgrid(x, y)

    fig = pylab.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=10, azim=60)
    surface = ax.plot_surface(x, y, df.values.T, cmap=cmap)
    cbar = fig.colorbar(surface, shrink=0.6, pad=0.1)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(15)
    ax.set_xlabel("Day to Maturity", labelpad=28, fontdict={"size": 15})
    ax.set_ylabel("Spot", labelpad=20, fontdict={"size": 15})
    ax.set_zlabel("Greeks (MC)", labelpad=24, fontdict={"size": 15})

    ax.tick_params(axis='x', labelsize=15, rotation=-40, pad=8)
    ax.tick_params(axis='y', labelsize=15, pad=12)
    ax.tick_params(axis='z', labelsize=15, pad=12)
    plt.show()
