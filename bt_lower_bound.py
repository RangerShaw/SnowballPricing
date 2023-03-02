import pandas as pd
import numpy as np
import openpyxl as op


class LowerBoundTester:

    def __init__(self, fp: str):
        self.bt_map = {
            '价差': self.bt_plain,
            '两区间': self.bt_plain,
            '三区间': self.bt_plain,
            '鲨鱼鳍': self.bt_shark_fin,
            '触碰式': self.bt_touch,
            '区间累计': self.bt_range_accrual,
            '自动敲出': self.bt_auto_call,
            '自动敲入敲出': self.bt_snowball,
        }
        self.products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
        self.closes_df = pd.read_excel(fp, sheet_name='历史走势')
        self.interval_map = self.build_intervals()
        self.period_months = np.array(list(self.interval_map.keys()))

    def build_intervals(self):
        interval_map = {}
        period_months = [0.25, 0.5, 1, 2, 3, 6, 9, 12, 24, 36]
        back_years = [0.25, 0.5, 1, 2, 3, 5, 9, 10, 12, 13]
        date_sr = self.closes_df['日期']
        today = date_sr.iloc[-1]
        print(f'today: {today}')

        for i, month in enumerate(period_months):
            period = pd.DateOffset(months=month) if month >= 1 else pd.DateOffset(days=round(28 * month))
            year = back_years[i]
            fst_sdate = today - (pd.DateOffset(years=year) if year >= 1 else pd.DateOffset(months=round(year * 12)))
            lst_sdate = today - period
            i_fst_sdate = (date_sr >= fst_sdate).argmax()
            i_lst_sdate = (date_sr <= lst_sdate).argmin()
            intervals = np.arange(i_fst_sdate, i_lst_sdate)
            intervals.resize((2, len(intervals)), refcheck=False)

            edates = date_sr.iloc[intervals[0]] + period
            bools = date_sr.values >= edates.values[:, None]
            intervals[1] = np.argmax(bools, axis=1)
            print(len(intervals[0]))
            interval_map[month] = intervals

        return interval_map

    def get_intervals(self, period, prices: np.ndarray):
        idx = np.abs(self.period_months - period).argmin()
        period = self.period_months[idx]  # nearest period
        intervals = self.interval_map[period]
        n_nan = np.isnan(prices[intervals[0]]).argmin()
        return intervals[:, n_nan:]

    def get_ratios(self, product: pd.Series):
        prices = self.closes_df[product['标的']].values
        intervals = self.get_intervals(product['期限(月)'], prices)
        ratios = prices[intervals[1]] / prices[intervals[0]]
        return ratios

    def bt_plain(self, product):
        ratios = self.get_ratios(product)
        low_bound = product['下端收益触达线']
        n_lower = np.sum(ratios < low_bound if product['方向'] == '看涨' else ratios > low_bound)
        return n_lower / (len(ratios))

    def bt_shark_fin(self, product):
        if product['方向'] == '两边':
            return None
        ratios = self.get_ratios(product)
        low_bound = product['下端收益触达线']
        n_lower = np.sum(ratios < low_bound if product['方向'] == '看涨' else ratios > low_bound)
        return n_lower / len(ratios)

    def bt_range_accrual(self, product):
        ratios = self.get_ratios(product)
        price_range = [float(x.replace('%', 'e-2')) for x in str(product['下端收益触达线']).split('；')]
        n_lower = np.sum((ratios < price_range[0]) | (ratios > price_range[1]))
        return n_lower / len(ratios)

    def bt_auto_call(self, product):
        prices = self.closes_df[product['标的']].values
        intervals = self.get_intervals(product['期限(月)'], prices)

        i_odates = np.linspace(intervals[0], intervals[1], 13, axis=1).round().astype(int)
        paths = prices[i_odates[:, 1:]]
        kout_prices = prices[i_odates[:, 0], None] * product['下端收益触达线']
        nkout_bools = paths < kout_prices if product['方向'] == '看涨' else paths > kout_prices
        n_nkout = np.all(nkout_bools, axis=1).sum()
        return n_nkout / len(intervals[0])

    def bt_snowball(self, product):
        is_call = product['方向'] == '看涨'
        prices = self.closes_df[product['标的']].values
        intervals = self.get_intervals(product['期限(月)'], prices)

        i_odates = np.linspace(intervals[0], intervals[1], 13, axis=1).round().astype(int)
        paths = prices[i_odates[:, 1:]]
        kout_prices = prices[i_odates[:, 0], None] * 1.0
        nkout_bool_matrix = paths < kout_prices if is_call else paths > kout_prices
        idx_nkout = np.all(nkout_bool_matrix, axis=1).nonzero()[0]

        kin_prices = prices[i_odates[:, 0], None] * product['下端收益触达线']
        n_kin_nkout = 0
        for i in idx_nkout:
            i_prices = prices[intervals[0][i]:intervals[1][i]]
            if np.any(i_prices < kin_prices[i] if is_call else i_prices > kin_prices[i]):
                n_kin_nkout += 1
        return n_kin_nkout / len(intervals[0])

    def bt_touch(self, product):
        prices = self.closes_df[product['标的']].values
        intervals = self.get_intervals(product['期限(月)'], prices)
        is_call = product['方向'] == '看涨' or product['方向'] == '触入看涨'
        low_prices = product['下端收益触达线'] * prices[intervals[0]]

        n_low = 0
        for i, interval in enumerate(intervals.T):
            i_prices = prices[interval[0]:interval[1]]
            if np.all(i_prices < low_prices[i] if is_call else i_prices > low_prices[i]):
                n_low += 1
        return n_low / len(intervals[0])

    def _backtest(self, product):
        if product['结构'] not in self.bt_map or product['标的'] not in self.closes_df.columns:
            return None
        # return self.bt_map[product['结构']](product)
        try:
            return self.bt_map[product['结构']](product)
        except:
            return None

    def run(self):
        return self.products.apply(self._backtest, axis=1)


if __name__ == '__main__':
    fp = 'D:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0303\\结构性产品同业发行结构汇总20230301.xlsx'
    tester = LowerBoundTester(fp)
    results = tester.run()
    print(results)

    # wb = op.load_workbook(fp)
    # ws = wb['历史发行结构汇总']
    # for r_idx, res in enumerate(results.tolist(), 4):
    #     ws.cell(row=r_idx, column=15, value=res)
    # wb.save(fp)

    # xw.Book(fp).sheets['历史发行结构汇总']['O4'].options(index=False, header=False).value = results
