import pandas as pd
import numpy as np
import openpyxl as op


class LowerBoundTester:

    def __init__(self, fp):
        self.bt_map = {
            '两区间': self.bt_plain,
            '三区间': self.bt_plain,
            '价差': self.bt_plain,
            '触碰式': self.bt_plain,
            '鲨鱼鳍': self.bt_shark_fin,
            '区间累计': self.bt_range_accrual,
            '自动敲出': self.bt_auto_call,
        }
        self.products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
        self.closes_df = pd.read_excel(fp, sheet_name='历史走势')
        self.products['期限(月)'] = np.floor(self.products['期限(月)'])
        self.interval_map = self.gen_intervals()

    def gen_intervals(self):
        interval_map = {}
        period_months = [0, 1, 2, 3, 6, 9, 12, 36]  # 0 is 14d
        back_years = [0.5, 1, 2, 3, 5, 9, 10, 13]
        date_sr = self.closes_df['日期']
        today = date_sr.iloc[-1]
        print(f'today: {today}')

        for i, month in enumerate(period_months):
            period = pd.DateOffset(months=month) if i != 0 else pd.DateOffset(days=14)

            fst_sdate = today - (pd.DateOffset(years=back_years[i]) if i != 0 else pd.DateOffset(months=6))
            lst_sdate = today - period
            i_fst_sdate = (date_sr >= fst_sdate).argmax()
            i_lst_sdate = (date_sr <= lst_sdate).argmin()
            intervals = np.arange(i_fst_sdate, i_lst_sdate)
            intervals.resize((2, len(intervals)), refcheck=False)

            edates = date_sr.iloc[intervals[0]] + period
            edates = edates.values.reshape(len(edates), 1)
            bools = date_sr.values >= edates
            intervals[1] = np.argmax(bools, axis=1)
            print(len(intervals[0]))
            interval_map[period_months[i]] = intervals

        return interval_map

    def get_ratios(self, product: pd.Series):
        intervals = self.interval_map[np.floor(product['期限(月)'])]
        prices = self.closes_df[product['标的']].values
        prices0 = prices[intervals[0]]
        ratios = prices[intervals[1]] / prices0
        return ratios, prices0

    def bt_auto_call(self, product):
        period = np.floor(product['期限(月)'])
        intervals = self.interval_map[period]
        prices = self.closes_df[product['标的']].values

        i_odates = np.linspace(intervals[0], intervals[1], 13, axis=1).round().astype(int)
        paths = prices[i_odates[:, 1:]]
        kout_prices = (prices[i_odates[:, 0]] * product['下端收益触达线']).reshape(paths.shape[0], 1)

        nkout_bools = paths < kout_prices if product['方向'] == '看涨' else paths > kout_prices
        n_nkout = np.all(nkout_bools, axis=1).sum()
        return n_nkout / len(intervals[0])

    def bt_plain(self, product):
        ratios, prices0 = self.get_ratios(product)
        low_bound = product['下端收益触达线']
        n_lower = np.sum(ratios < low_bound if product['方向'] == '看涨' else ratios > low_bound)
        return n_lower / (len(ratios) - np.isnan(prices0).sum())  # 民生全球

    def bt_shark_fin(self, product):
        if product['方向'] == '两边':
            return None
        ratios, _ = self.get_ratios(product)
        low_bound = product['下端收益触达线']
        n_lower = np.sum(ratios < low_bound if product['方向'] == '看涨' else ratios > low_bound)
        return n_lower / len(ratios)

    def bt_range_accrual(self, product):
        ratios, _ = self.get_ratios(product)
        price_range = [float(x.replace('%', 'e-2')) for x in str(product['下端收益触达线']).split('；')]
        n_lower = np.sum((ratios < price_range[0]) | (ratios > price_range[1]))
        return n_lower / len(ratios)

    def backtest(self, product):
        if not (product['结构'] in self.bt_map and product['标的'] in self.closes_df.columns
                and product['期限(月)'] in self.interval_map):
            return None
        else:
            return self.bt_map[product['结构']](product)

    def run(self):
        return self.products.apply(self.backtest, axis=1)


if __name__ == '__main__':
    fp = 'G:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0224\\结构性产品同业发行结构汇总20230222.xlsx'
    tester = LowerBoundTester(fp)
    results = tester.run()
    print(results)

    # xw.Book(fp).sheets['历史发行结构汇总']['O4'].options(index=False, header=False).value = results
    # wb = op.load_workbook(fp)
    # ws = wb['历史发行结构汇总']
    # for r_idx, res in enumerate(results.tolist(), 4):
    #     ws.cell(row=r_idx, column=15, value=res)
    # wb.save(fp)
