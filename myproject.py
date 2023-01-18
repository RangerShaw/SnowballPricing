# -*- coding: utf-8 -*-
import xlwings as xw
import numpy as np

N_DAYS_A_YEAR = 365
TRADING_DAYS_A_YEAR = 252  # delta t in MC path
N_MC_PATHS = 100000
FLUCTUATION_LIMIT = 0  # 0: no limit


class SmallsbM2M:

    def __init__(self, trading_days):
        self.trade_dates = np.array(trading_days)
        self.date_map = {v: index[0] for index, v in np.ndenumerate(self.trade_dates)}

    def gen_trends_mc(self, days, S0, r, q, v):
        rand = np.random.randn(N_MC_PATHS, days)
        mu = r - q
        d_t = 1.0 / TRADING_DAYS_A_YEAR
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

    def m2m_365(self, principle, s1, s_kout, funding, coupon_rate, r, q, v, value_date, obs_dates, s_date, e_date):
        op_days = (value_date - s_date).days
        left_days = (e_date - value_date).days

        # knock out immediately
        if value_date in obs_dates and s1 >= s_kout:
            return (coupon_rate - funding) * principle * op_days / N_DAYS_A_YEAR

        # no more observe day
        obs_dates = obs_dates[obs_dates > value_date]
        if len(obs_dates) == 0:
            return -funding * principle * np.exp(-r * left_days / N_DAYS_A_YEAR)

        value_date_t = self.date_map[value_date]
        e_date_t = self.date_map[e_date]
        obs_dates_t = np.array([self.date_map[d] for d in obs_dates]) - value_date_t
        kout_days_map = np.array([d.days for d in obs_dates - s_date])

        paths = self.gen_trends_mc(e_date_t - value_date_t, s1, r, q, v)
        kout_dates_index_t = self.classify_path(paths, obs_dates_t, s_kout)
        kout_periods_yr = kout_days_map[kout_dates_index_t] / N_DAYS_A_YEAR

        pv = self.valuate(principle, coupon_rate, r, funding, kout_periods_yr, op_days, left_days)
        return pv


def check_paras(trading_days, value_date, s_date, e_date, obs_dates):
    tday_set = set(trading_days)
    return value_date in tday_set and e_date in tday_set and all(d in tday_set for d in obs_dates)


def m2m():
    wb = xw.Book.caller()
    trading_days = wb.sheets['trading_days'].range('A1').options(transpose=True, expand='table').value

    sheet = wb.sheets[0]
    obs_dates = sheet.range('B12').options(transpose=True, expand='down').value
    market_paras = sheet['C3:C7'].value
    product_paras = sheet['F3:F8'].value
    [value_date, S0, v, r, q] = market_paras
    [s_date, e_date, funding, coupon_rate, s_kout, principal] = product_paras

    if not check_paras(trading_days, value_date, s_date, e_date, obs_dates):
        sheet["G3"].value = '有日期为非交易日'
        return

    sb365 = SmallsbM2M(trading_days)
    pv1 = sb365.m2m_365(principal, S0, s_kout, funding, coupon_rate, r, q, v, value_date, np.array(obs_dates), s_date,
                        e_date)
    print(pv1)
    sheet["G3"].value = pv1
    # input()


def m2m_batch():
    wb = xw.Book.caller()
    sheet = wb.sheets['Batch']
    trading_days = wb.sheets['trading_days'].range('A1').options(transpose=True, expand='down').value


if __name__ == "__main__":
    pass
