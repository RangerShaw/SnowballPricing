import pandas as pd
import numpy as np
import openpyxl as op

interval_map = {}
structs = {'两区间', '三区间', '鲨鱼鳍', '价差', '触碰式', '区间累计', '自动敲出'}
underlyings = []


def gen_intervals(closes_df: pd.DataFrame):
    period_months = [0, 1, 2, 3, 6, 9, 12, 36]  # 0 is 14d
    back_years = [0.5, 1, 2, 3, 5, 9, 10, 13]
    date_sr = closes_df['日期']
    today = date_sr.iloc[-1]
    print(f'today: {today}')

    for i, month in enumerate(period_months):
        intervals = [None, None]
        period = pd.DateOffset(months=month) if i != 0 else pd.DateOffset(days=14)

        fst_sdate = today - (pd.DateOffset(years=back_years[i]) if i != 0 else pd.DateOffset(months=6))
        lst_sdate = today - period
        i_fst_sdate = (date_sr >= fst_sdate).argmax()
        i_lst_sdate = (date_sr <= lst_sdate).argmin()
        intervals[0] = np.arange(i_fst_sdate, i_lst_sdate)

        edates = date_sr.iloc[intervals[0]] + period
        edates = edates.values.reshape(len(edates), 1)
        bools = date_sr.values >= edates
        intervals[1] = np.argmax(bools, axis=1)

        print(len(intervals[0]))
        interval_map[period_months[i]] = intervals


def auto_call(closes_df: pd.DataFrame, product, call: bool):
    period = np.floor(product['期限(月)'])
    intervals = interval_map[period]
    prices = closes_df[product['标的']].values

    i_odates = np.linspace(intervals[0], intervals[1], 13, axis=1).round().astype(int)
    paths = prices[i_odates[:, 1:]]
    kout_prices = (prices[i_odates[:, 0]] * product['下端收益触达线']).reshape(paths.shape[0], 1)

    nkout_bools = paths < kout_prices if call else paths > kout_prices
    n_nkout = np.all(nkout_bools, axis=1).sum()
    return n_nkout / len(intervals[0])


def bt_plain(closes_df, product):
    period, low_bound = np.floor(product['期限(月)']), product['下端收益触达线']
    intervals = interval_map[period]
    prices = closes_df[product['标的']].values
    changes = prices[intervals[1]] / prices[intervals[0]]
    bools = changes < low_bound if product['方向'] == '看涨' else changes > low_bound
    n_lower = np.sum(changes < low_bound if product['方向'] == '看涨' else changes > low_bound)
    nnull = closes_df[product['标的']].isnull().sum()
    return n_lower / (len(changes) - closes_df[product['标的']].isnull().sum())


def backtest(closes_df, product):
    period = np.floor(product['期限(月)'])
    intervals = interval_map[period]
    prices0 = closes_df[product['标的']].values[intervals[0]]
    prices1 = closes_df[product['标的']].values[intervals[1]]
    changes = prices1 / prices0
    struct, low_bound, direction = product['结构'], product['下端收益触达线'], product['方向']

    if struct == '两区间' or struct == '三区间' or struct == '价差' or struct == '触碰式':
        return bt_plain(closes_df, product)
    elif struct == '鲨鱼鳍':
        if direction == '两边':
            return None
        else:
            n_lower = np.sum(changes < low_bound if direction == '看涨' else changes > low_bound)
        return n_lower / len(changes)
    elif struct == '区间累计':
        lows = [float(x.replace('%', 'e-2')) for x in str(low_bound).split('；')]
        n_lower = np.sum((changes < lows[0]) | (changes > lows[1]))
        return n_lower / len(changes)
    elif struct == '自动敲出':
        return auto_call(closes_df, product, direction == '看涨')
    return None


def cal_lower_p(product, closes_df):
    if product['结构'] not in structs or product['标的'] not in underlyings or product['期限(月)'] not in interval_map:
        return None
    else:
        return backtest(closes_df, product)


if __name__ == '__main__':
    fp = 'D:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0224\\结构性产品同业发行结构汇总20230222.xlsx'
    products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
    closes_df = pd.read_excel(fp, sheet_name='历史走势')

    products['期限(月)'] = np.floor(products['期限(月)'])
    underlyings = set(closes_df.columns)
    gen_intervals(closes_df)

    results = products.apply(cal_lower_p, closes_df=closes_df, axis=1)
    print(results)

    # xw.Book(fp).sheets['历史发行结构汇总']['O4'].options(index=False, header=False).value = results
    # wb = op.load_workbook(fp)
    # ws = wb['历史发行结构汇总']
    # for r_idx, res in enumerate(results.tolist(), 4):
    #     ws.cell(row=r_idx, column=15, value=res)
    # wb.save(fp)
