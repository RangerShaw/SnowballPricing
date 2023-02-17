import math

import pandas as pd
import numpy as np

fp = 'D:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0215\\结构性产品同业发行结构汇总20230215.xlsx'
products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
closes_df = pd.read_excel(fp, sheet_name='历史走势')


def gen_periods():
    pass


def touch():
    pass


def spread():
    pass


struct_map = {
    '触碰式': gen_periods(),
    '价差': spread(),
}

valid_structures = {'两区间', '三区间', '鲨鱼鳍', '价差', '区间累计', '自动敲出'}

interval_map = {}


def backtest(closes_df, product):
    period = math.floor(product['期限'])
    intervals = interval_map[period]
    prices0 = closes_df[product['标的']].values[intervals[0]]
    prices1 = closes_df[product['标的']].values[intervals[1]]
    changes = prices1 / prices0
    n_lower = 0

    struct = product['结构']
    if struct == '两区间' or struct == '三区间' or struct == '价差' or struct == '自动敲出':
        n_lower = np.sum(changes < product['下端收益触达线'] if product['方向'] == '看涨' else changes > product['下端收益触达线'])
        return n_lower / len(prices0)
    elif struct == '鲨鱼鳍':
        if product['方向'] == '两边':
            lows = str(product['下端收益触达线']).split('；')
            n_lower = np.sum(float(lows[0]) < changes < float(lows[1]))
        else:
            n_lower = np.sum(changes < product['下端收益触达线'] if product['方向'] == '看涨' else changes > product['下端收益触达线'])
        return n_lower / len(prices0)
    elif struct == '区间累计':
        lows = str(product['下端收益触达线']).split('；')
        n_lower = np.sum((changes < lows[0]) | (changes > lows[1]))
        return n_lower / len(prices0)
    return None


if __name__ == '__main__':
    fp = 'D:\\OneDrive\\Intern\\CIB\\Work\\同业结构\\0215\\结构性产品同业发行结构汇总20230215.xlsx'
    products = pd.read_excel(fp, sheet_name='历史发行结构汇总', usecols='H:M', skiprows=2)
    closes_df = pd.read_excel(fp, sheet_name='历史走势')
    print(products.head(2))
    print(closes_df.head(2))
    close_set = set(closes_df.columns)
    print(close_set)
    gen_periods()

    res = []
    # for index, product in products.iterrows():
    #     if product['结构'] not in valid_structures or product['标的'] not in close_set:
    #         res.append(None)
    #     else:
    #         res.append(backtest(closes_df, product))
    print(res)
