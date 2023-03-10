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
        self.periods = [(0.25, 0.25), (0.5, 0.5), (1, 1), (2, 2), (3, 3), (6, 5), (9, 9), (12, 10), (24, 10), (36, 10)]
        self.products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
        self.closes_df = pd.read_excel(fp, sheet_name='历史走势')
        self.interval_map = self.build_intervals()
        self.period_months = np.array(list(self.interval_map.keys()))

    def build_intervals(self):
        interval_map = {}
        date_sr = self.closes_df['日期']
        today = date_sr.iloc[-1]
        print(f'today: {today}')

        for i, (month, year) in enumerate(self.periods):
            period = pd.DateOffset(months=month) if month >= 1 else pd.DateOffset(days=round(28 * month))
            fst_sdate = today - (pd.DateOffset(years=year) if year >= 1 else pd.DateOffset(months=round(year * 12)))
            lst_sdate = today - period
            i_fst_sdate = (date_sr >= fst_sdate).argmax()
            i_lst_sdate = (date_sr <= lst_sdate).argmin()
            intervals = np.arange(i_fst_sdate, i_lst_sdate)
            intervals.resize((2, len(intervals)), refcheck=False)

            edates = date_sr.iloc[intervals[0]] + period
            bools = date_sr.values >= edates.values[:, None]
            intervals[1] = np.argmax(bools, axis=1)
            interval_map[month] = intervals
            print(len(intervals[0]))

        return interval_map

    def get_prices_intervals(self, underlying, period):
        prices = self.closes_df[underlying].values
        if period not in self.interval_map:
            period = self.period_months[np.abs(self.period_months - period).argmin()]  # nearest period
        intervals = self.interval_map[period]
        n_nan = np.isnan(prices[intervals[0]]).argmin()
        return prices, intervals[:, n_nan:]

    def get_ratios(self, product: pd.Series):
        prices, intervals = self.get_prices_intervals(product['标的'], product['期限(月)'])
        ratios = prices[intervals[1]] / prices[intervals[0]]
        return ratios

    def bt_plain(self, product):
        ratios = self.get_ratios(product)
        low_bound = product['下端收益触达线']
        n_low = np.sum(ratios < low_bound if product['方向'] == '看涨' else ratios > low_bound)
        return n_low / (len(ratios))

    def bt_shark_fin(self, product):
        return None if product['方向'] == '两边' else self.bt_plain(product)

    def bt_range_accrual(self, product):
        ratios = self.get_ratios(product)
        price_range = [float(x.replace('%', 'e-2')) for x in str(product['下端收益触达线']).split('；')]
        n_low = np.sum((ratios < price_range[0]) | (ratios > price_range[1]))
        return n_low / len(ratios)

    def not_kout(self, prices, intervals, is_upside: bool, months, kout_p_ratio):
        i_odates = np.linspace(intervals[0], intervals[1], months + 1, axis=1).round().astype(int)
        paths = prices[i_odates[:, 1:]]
        kout_prices = prices[i_odates[:, 0], None] * kout_p_ratio
        nkout_bool_matrix = paths < kout_prices if is_upside else paths > kout_prices
        return np.all(nkout_bool_matrix, axis=1)

    def bt_auto_call(self, product):
        is_upside = product['方向'] == '看涨'
        prices, intervals = self.get_prices_intervals(product['标的'], product['期限(月)'])
        nkout_bool_array = self.not_kout(prices, intervals, is_upside, product['期限(月)'], product['下端收益触达线'])
        n_nkout = nkout_bool_array.sum()
        return n_nkout / len(intervals[0])

    def count_touch(self, prices, intervals, is_upside: bool, strike_p_ratio):
        strike_prices = prices[intervals[0]] * strike_p_ratio
        n_touch = 0
        for i, interval in enumerate(intervals.T):
            prices_i = prices[interval[0] + 1:interval[1]]
            if np.any(prices_i >= strike_prices[i] if is_upside else prices_i <= strike_prices[i]):
                n_touch += 1
        return n_touch

    def bt_touch(self, product):
        is_upside = product['方向'] == '看涨' or product['方向'] == '触入看涨'
        prices, intervals = self.get_prices_intervals(product['标的'], product['期限(月)'])
        n_touch = self.count_touch(prices, intervals, is_upside, product['下端收益触达线'])
        return (len(intervals[0]) - n_touch) / len(intervals[0])

    def bt_snowball(self, product):
        is_upside = product['方向'] == '看涨'
        prices, intervals = self.get_prices_intervals(product['标的'], product['期限(月)'])

        nkout_bool_array = self.not_kout(prices, intervals, is_upside, product['期限(月)'], 1.0)
        idx_nkout = nkout_bool_array.nonzero()[0]
        n_kin_nkout = self.count_touch(prices, intervals[:, idx_nkout], not is_upside, product['下端收益触达线'])
        return n_kin_nkout / len(intervals[0])

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
    fp = 'D:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0310\\结构性产品同业发行结构汇总20230308.xlsx'
    tester = LowerBoundTester(fp)
    results = tester.run()
    print(results)

    # wb = op.load_workbook(fp)
    # ws = wb['历史发行结构汇总']
    # for r_idx, res in enumerate(results.tolist(), 4):
    #     ws.cell(row=r_idx, column=15, value=res)
    # wb.save(fp)

    # xw.Book(fp).sheets['历史发行结构汇总']['O4'].options(index=False, header=False).value = results
