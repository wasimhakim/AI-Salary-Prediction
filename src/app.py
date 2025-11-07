import numpy as np
import pandas as pd

salary_data = pd.read_csv('./data/Experience-Salary.csv')

# first 800 rows for training data
experience = np.array(salary_data['exp(in months)'][:800])
salary = np.array(salary_data['salary(in thousands)'][:800])