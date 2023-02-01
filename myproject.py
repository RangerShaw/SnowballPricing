# -*- coding: utf-8 -*-
import datetime
import math

import xlwings as xw
import numpy as np
from dateutil.relativedelta import relativedelta

N_DAYS_A_YEAR = 365
N_TRADE_DAYS_A_YEAR = 252  # delta t in MC path
N_MC_PATHS = 100000
FLUCTUATION_LIMIT = 0.1  # 0: no limit


class SmallsbM2M:

    def __init__(self, trading_days):
        self.trade_dates = np.array(trading_days)
        self.tdate_set = set(trading_days)
        self.tdate_map = {v: index for index, v in enumerate(trading_days)}
        self.paths = None

    def gen_paths_mc(self, days, S0, r, q, v):
        rand = np.random.randn(N_MC_PATHS, days)
        mu = r - q
        d_t = 1.0 / N_TRADE_DAYS_A_YEAR
        d_ln_S = (mu - 0.5 * v ** 2) * d_t + v * np.sqrt(d_t) * rand
        d_ln_S = np.insert(d_ln_S, 0, values=np.zeros(rand.shape[0]), axis=1)  # S0
        if FLUCTUATION_LIMIT > 0:
            upper, lower = np.log(1 + FLUCTUATION_LIMIT), np.log(1 - FLUCTUATION_LIMIT)
            d_ln_S[d_ln_S > upper] = upper
            d_ln_S[d_ln_S < lower] = lower
        ln_S = np.cumsum(d_ln_S, axis=1)
        S_all = S0 * np.exp(ln_S)
        return S_all

    def classify_path(self, paths, kout_obs_days, s_kout):
        kout_bool_matrix = paths[:, kout_obs_days] >= s_kout
        kout_flags = np.any(kout_bool_matrix, axis=1)
        kout_days_index = kout_bool_matrix[kout_flags, :].argmax(axis=1) if kout_flags.any() else []
        return kout_days_index

    def valuate(self, principal, coupon_rate, r, funding, kout_periods_yr, op_days, left_days):
        # knock out
        kout_disc_periods_yr = kout_periods_yr - (op_days / N_DAYS_A_YEAR)
        kout_profits = principal * (coupon_rate - funding) * kout_periods_yr * np.exp(-r * kout_disc_periods_yr)
        pv_kout = np.sum(kout_profits)

        # not knock out
        kin_disc_period_yr = left_days / N_DAYS_A_YEAR
        nkout_losses = (N_MC_PATHS - len(kout_periods_yr)) * principal * funding * np.exp(-r * kin_disc_period_yr)
        return (pv_kout - nkout_losses) / N_MC_PATHS

    def m2m_365(self, principal, s1, s_kout, funding, cp_rate, r, q, v, value_date, obs_dates, s_date, e_date):
        op_days = (value_date - s_date).days
        left_days = (e_date - value_date).days

        if value_date in obs_dates and s1 >= s_kout:  # knock out immediately
            return (cp_rate - funding) * principal * op_days / N_DAYS_A_YEAR
        if np.all(obs_dates < value_date):  # no more observe day
            return -funding * principal * np.exp(-r * left_days / N_DAYS_A_YEAR)

        obs_dates = obs_dates[obs_dates > value_date]
        value_date_t = self.tdate_map[value_date]
        e_date_t = self.tdate_map[e_date]
        obs_dates_t = [self.tdate_map[d] - value_date_t for d in obs_dates]
        kout_periods_map = np.array([d.days / N_DAYS_A_YEAR for d in obs_dates - s_date])

        n_t_days = e_date_t - value_date_t
        paths = self.gen_paths_mc(n_t_days, s1, r, q, v) if self.paths is None else self.paths[:, :n_t_days + 1]
        kout_dates_index_t = self.classify_path(paths, obs_dates_t, s_kout)
        kout_periods_yr = kout_periods_map[kout_dates_index_t]

        pv = self.valuate(principal, cp_rate, r, funding, kout_periods_yr, op_days, left_days)
        return pv

    def check_paras(self, e_date, obs_dates):
        return e_date in self.tdate_set and all(d in self.tdate_set for d in obs_dates)

    def m2m_single_365(self, principal, s0, s_kout, funding, cp_rate, r, q, v, value_date, obs_dates, s_date, e_date):
        if not self.check_paras(e_date, obs_dates):
            return '有日期为非交易日'

        real_value_date = value_date
        while real_value_date not in self.tdate_set:
            real_value_date -= datetime.timedelta(days=1)

        pv = self.m2m_365(principal, s0, s_kout, funding, cp_rate, r, q, v, real_value_date, obs_dates, s_date, e_date)
        return pv * np.exp(r * (value_date - real_value_date).days / N_DAYS_A_YEAR)

    def gen_obs_dates(self, s_date, e_date):
        obs_dates = []

        i = 1
        obs_date = s_date + relativedelta(months=i)
        while obs_date <= e_date:
            while obs_date not in self.tdate_set:
                obs_date += datetime.timedelta(days=1)
            obs_dates.append(obs_date)
            i += 1
            obs_date = s_date + relativedelta(months=i)
        return np.array(obs_dates)

    def gen_batch_paths(self, s0, r, q, v, value_date, products_paras):
        latest_date = max(prod[1] for prod in products_paras)
        while latest_date not in self.tdate_set:
            latest_date += datetime.timedelta(days=1)
        return self.gen_paths_mc(self.tdate_map[latest_date] - self.tdate_map[value_date], s0, r, q, v)

    def m2m_batch_365(self, s0, r, q, v, value_date, products_paras):
        real_v_date = value_date
        while real_v_date not in self.tdate_set:
            real_v_date -= datetime.timedelta(days=1)
        self.paths = self.gen_batch_paths(s0, r, q, v, real_v_date, products_paras)

        pvs = []
        for product in products_paras:
            [s_date, e_date, funding, cp_rate, s_kout, prin, status] = product
            if status == '否':
                pvs.append(None)
            elif e_date not in self.tdate_set:
                pvs.append('有日期为非交易日')
            else:
                obs_dates = self.gen_obs_dates(s_date, e_date)
                pv = self.m2m_365(prin, s0, s_kout, funding, cp_rate, r, q, v, real_v_date, obs_dates, s_date, e_date)
                pvs.append(pv * math.exp(r * (value_date - real_v_date).days / N_DAYS_A_YEAR))
            print(pvs[-1])
        return pvs


RESULT_CELL = 'G3'
BATCH_RESULT_CELL = 'K16'


def m2m(sheet_name):
    wb = xw.Book.caller()
    sheet = wb.sheets[sheet_name]

    trading_days = wb.sheets['trading_days'].range('A1').expand('down').value
    obs_dates = np.array(sheet.range('B12').expand('down').value)
    [value_date, S0, v, r, q] = sheet['C3:C7'].value
    [s_date, e_date, funding, cp_rate, s_kout, principal] = sheet['F3:F8'].value

    sb365 = SmallsbM2M(trading_days)
    pv = sb365.m2m_single_365(principal, S0, s_kout, funding, cp_rate, r, q, v, value_date, obs_dates, s_date, e_date)
    print(pv)
    sheet[RESULT_CELL].value = pv


def m2m_batch(sheet_name):
    wb = xw.Book.caller()
    sheet = wb.sheets[sheet_name]

    [value_date, S0, v, r, q] = sheet['C3:C7'].value
    products_paras = sheet['C16:I16'].expand('down').value
    trading_days = wb.sheets['trading_days'].range('A1').expand('down').value

    sb365 = SmallsbM2M(trading_days)
    pvs = sb365.m2m_batch_365(S0, r, q, v, value_date, products_paras)
    sheet[BATCH_RESULT_CELL].options(transpose=True).value = pvs


def main():
    pass
