# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:55:41 2019

@author: Administrator@CPC
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import copy


class Vanilla:
    def __init__(self, _s, _k, _r, _q, _sigma, _t, _typeflag, _Ndays1year=252, _timetype='years'):
        """
        -------------------------------------------------
        _typeflag：期权类型，提供以下2类：
            1) 'c'=call,    2) 'p'=put
        _timetpye：输入时间参数的单位，提供两个参数：years与days
            years代表单位为年，days代表单位为交易日。每年252个交易日。

        """
        self.s = _s
        self.k = _k
        self.r = _r
        self.q = _q  # dividend
        self.v = _sigma
        self.typeflag = _typeflag
        self.timetype = _timetype
        self.Ndays1year = _Ndays1year

        if self.timetype == 'days':
            self.t = _t / 252
        elif self.timetype == 'years':
            self.t = _t
        else:
            raise (Exception('_timetpye目前仅提供两个参数可选：years与days'))

    def valuation(self, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: value = Vanilla.valuation(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同
        '''
        try:
            s = kwargs['s']
        except:
            s = self.s

        try:
            v = kwargs['v']
        except:
            v = self.v

        try:
            t = kwargs['t']
            if self.timetype == 'days':
                t = t / 252
        except:
            t = self.t

        d1 = (np.log(s / self.k) + (self.r - self.q + v ** 2 / 2) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)

        if self.typeflag == 'c':
            value = s * np.exp(-self.q * t) * norm.cdf(d1) - self.k * np.exp(-self.r * t) * norm.cdf(d2)
        elif self.typeflag == 'p':
            value = -s * np.exp(-self.q * t) * norm.cdf(-d1) + self.k * np.exp(-self.r * t) * norm.cdf(-d2)
        else:
            raise (Exception('_typeflag目前提供2个参数可选：c、p'))

        return value

    def delta(self, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: delta = Vanilla.delta(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同

        '''
        try:
            s_greek = kwargs['s']
        except:
            s_greek = self.s

        try:
            v_greek = kwargs['v']
        except:
            v_greek = self.v

        try:
            t_greek = kwargs['t']
            if self.timetype == 'days':
                t_greek = t_greek / 252
        except:
            t_greek = self.t

        d1 = (np.log(s_greek / self.k) + (self.r - self.q + v_greek ** 2 / 2) * t_greek) / (v_greek * np.sqrt(t_greek))

        if self.typeflag == 'c':
            delta = np.exp(-self.q * t_greek) * norm.cdf(d1)
        elif self.typeflag == 'p':
            delta = -np.exp(-self.q * t_greek) * norm.cdf(-d1)
        else:
            raise (Exception('_typeflag目前提供2个参数可选：c、p'))

        return delta

    def gamma(self, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: gamma = Vanilla.gamma(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同


        '''
        try:
            s_greek = kwargs['s']
        except:
            s_greek = self.s

        try:
            v_greek = kwargs['v']
        except:
            v_greek = self.v

        try:
            t_greek = kwargs['t']
            if self.timetype == 'days':
                t_greek = t_greek / 252
        except:
            t_greek = self.t

        d1 = (np.log(s_greek / self.k) + (self.r - self.q + v_greek ** 2 / 2) * t_greek) / (v_greek * np.sqrt(t_greek))
        gamma = np.exp(-self.q * t_greek) * norm.pdf(d1) / (s_greek * v_greek * np.sqrt(t_greek))

        return gamma

    def vega(self, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: vega = Vanilla.vega(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同


        '''
        try:
            s_greek = kwargs['s']
        except:
            s_greek = self.s

        try:
            v_greek = kwargs['v']
        except:
            v_greek = self.v

        try:
            t_greek = kwargs['t']
            if self.timetype == 'days':
                t_greek = t_greek / 252
        except:
            t_greek = self.t

        d1 = (np.log(s_greek / self.k) + (self.r - self.q + v_greek ** 2 / 2) * t_greek) / (v_greek * np.sqrt(t_greek))
        vega = s_greek * np.exp(-self.q * t_greek) * norm.pdf(d1) * np.sqrt(t_greek)

        return vega

    def theta(self, **kwargs):
        '''
        -------------------------------------------------
        提供3个可变参数：s、v、t，对于没有指定的参数，将使用定义类时确定的参数
        t的类型与定义类时使用的相同
        eg: theta = Vanilla.theta(s=np.array([0.9,1,1.1]), v=0.21)
        若指定参数中有向量，则向量的长度需相同
        '''
        try:
            s_greek = kwargs['s']
        except:
            s_greek = self.s

        try:
            v_greek = kwargs['v']
        except:
            v_greek = self.v

        try:
            t_greek = kwargs['t']
            if self.timetype == 'days':
                t_greek = t_greek / 252
        except:
            t_greek = self.t

        d1 = (np.log(s_greek / self.k) + (self.r - self.q + v_greek ** 2 / 2) * t_greek) / (v_greek * np.sqrt(t_greek))
        d2 = d1 - v_greek * np.sqrt(t_greek)

        if self.typeflag == 'c':
            theta = -s_greek * np.exp(-self.q * t_greek) * norm.pdf(d1) * v_greek / (2 * np.sqrt(t_greek)) - \
                    self.r * self.k * np.exp(-self.r * t_greek) * norm.cdf(d2) + self.q * s_greek * np.exp(
                -self.q * t_greek) * norm.cdf(d1)
        elif self.typeflag == 'p':
            theta = -s_greek * np.exp(-self.q * t_greek) * norm.pdf(d1) * v_greek / (2 * np.sqrt(t_greek)) + \
                    self.r * self.k * np.exp(-self.r * t_greek) * norm.cdf(-d2) - self.q * s_greek * np.exp(
                -self.q * t_greek) * norm.cdf(-d1)
        else:
            raise (Exception('_typeflag目前提供2个参数可选：c、p'))

        return theta

    def QuasiRandSeed(self, filename, MC_lens, T_lens):
        """
        ---------------------------------------------------------
        此函数用于使用外部文件中定义的随机数种子
        """
        QuasiRand = np.array(pd.read_pickle(filename))
        if MC_lens > len(QuasiRand):
            print(" MC length is too long!")
        RandSeed = QuasiRand[:MC_lens, :T_lens]
        return RandSeed

    def MonteCarloGenerate(self, St, filename, MC_lens, T_lens, MCMethod="Sobol"):
        """
        ---------------------------------------------------------
        St是t=0时的价格

        此函数用于使用MC方法生成模拟序列
        MC方法可以选择"Sobol"或其他，使用Sobol方法需要给出对应的种子文件地址
        若使用普通方法，filename和MCMethod参数可以随意输入

        """
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

    def MCSolver(self, SAll):
        '''
        ---------------------------------------------------------
        此函数用于使用MC方法计算期权估值
        SAll：已有的模拟序列

        '''
        [SM, SN] = SAll.shape

        OutPut = pd.DataFrame(np.zeros([SM, 2]), columns=['OptionPrice', 'LastPrice'])
        LastPrice = copy.deepcopy(SAll[:, -1])
        OptionPrice = copy.deepcopy(SAll[:, -1]) - self.k
        if self.typeflag == 'c':
            OptionPrice[OptionPrice < 0] = 0
        elif self.typeflag == 'p':
            OptionPrice[OptionPrice > 0] = 0
            OptionPrice = - OptionPrice

        OutPut['OptionPrice'] = OptionPrice * np.exp(-self.r * self.t)
        OutPut['LastPrice'] = LastPrice

        return OutPut
