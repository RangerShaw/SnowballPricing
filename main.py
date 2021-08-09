# -*- coding: utf-8 -*-
from snowball import SnowBall


if __name__ == '__main__':
    fp = 'data/QuasiRand.pickle'
    sb = SnowBall(1.0, 0.036, 0.3, 1, 0.75, 1.0)
    sb.solve_mc(fp)
