import numpy as np
import pandas as pd
import math

def students_gddscnt(x,y):
    mval = bval = 0
    iter = 1000
    n = len(x)
    lenrate = 0.001
    cost_prev = 0

    for i in range(iter):
        y_pred = mval * x + bval

        cost = (1/n) * sum([val**2 for val in (y - y_pred)])

        md = -(2/n)* sum(x*(y - y_pred))
        bd = -(2/n)* sum(y-y_pred)

        mval = mval - lenrate * md
        bval = bval - lenrate * bd

        if math.isclose(cost, cost_prev, rel_tol=1e-20):
            break
        cost_prev=cost

        print("mval {}, bval {}, cost {}, iterations {}".format(mval, bval, cost, i))

df = pd.read_csv("test_scores.csv")
x = np.array(df.math)
y = np.array(df.cs)

students_gddscnt(x,y)