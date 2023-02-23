import pandas as pd
import numpy as np
import openpyxl as op

interval_map = {}
structs = {'两区间', '三区间', '鲨鱼鳍', '价差', '触碰式', '区间累计', '自动敲出'}
underlyings = []


def gen_intervals(closes_df: pd.DataFrame):
    period_months = [0, 1, 2, 3, 6, 9, 12]  # 0 is 14d
    back_years = [0.5, 1, 2, 3, 5, 9, 10]
    date_df = closes_df['日期']
    today = date_df.iloc[-1]
    print(f'today: {today}')

    for i, month in enumerate(period_months):
        intervals = [None, None]
        sdate = pd.Timestamp(today - (pd.DateOffset(years=back_years[i]) if i != 0 else pd.DateOffset(months=6)))
        i_sdate = (closes_df['日期'] >= sdate).argmax()
        intervals[0] = np.arange(i_sdate, len(closes_df))

        intervals[1] = intervals[0].copy()
        for j in range(len(intervals[1])):
            i_s = intervals[0][j]
            edate = date_df.iloc[intervals[0][j]] + (pd.DateOffset(months=month) if i != 0 else pd.DateOffset(days=14))
            bools = closes_df.loc[i_s:, '日期'] >= edate
            if bools.any():
                intervals[1][j] = bools.idxmax()
            else:
                intervals[0].resize(j, refcheck=False)
                intervals[1].resize(j, refcheck=False)
                break
        print(len(intervals[0]))
        interval_map[period_months[i]] = intervals


def auto_call(closes_df, product):

    pass


def backtest(closes_df, product):
    period = np.floor(product['期限(月)'])
    intervals = interval_map[period]
    prices0 = closes_df[product['标的']].values[intervals[0]]
    prices1 = closes_df[product['标的']].values[intervals[1]]
    changes = prices1 / prices0
    struct, low_bound = product['结构'], product['下端收益触达线']

    if struct == '两区间' or struct == '三区间' or struct == '价差' or struct == '触碰式':
        n_lower = np.sum(changes < low_bound if product['方向'] == '看涨' else changes > low_bound)
        return n_lower / len(changes)
    elif struct == '鲨鱼鳍':
        if product['方向'] == '两边':
            return None
            # lows = str(low_bound).split('；')
            # n_lower = np.sum((float(lows[0]) < changes) & (changes < float(lows[1])))
        else:
            n_lower = np.sum(changes < low_bound if product['方向'] == '看涨' else changes > low_bound)
        return n_lower / len(changes)
    elif struct == '区间累计':
        lows = [float(x.replace('%', 'e-2')) for x in str(low_bound).split('；')]
        n_lower = np.sum((changes < lows[0]) | (changes > lows[1]))
        return n_lower / len(changes)
    elif struct == '自动敲出':
        return None
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
    wb = op.load_workbook(fp)
    ws = wb['历史发行结构汇总']
    for r_idx, res in enumerate(results.tolist(), 4):
        ws.cell(row=r_idx, column=15, value=res)
    wb.save(fp)
