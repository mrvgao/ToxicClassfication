import pandas as pd


test_result = pd.read_csv('/Users/Minchiuan/Downloads/test_result.csv')

eps = 1.95313e-08
zero = 0.0

test_result = test_result.replace(eps, zero)
print(test_result.head())
test_result.to_csv('cust_data/submission.csv', index=False)


print('write end!')
