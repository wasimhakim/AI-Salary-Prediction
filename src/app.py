import numpy as np
import pandas as pd

from brain import Brain

salary_data = pd.read_csv('./data/Experience-Salary.csv')

# first 900 rows for training data
experience = np.array(salary_data['exp(in months)'][:900])
salary = np.array(salary_data['salary(in thousands)'][:900])

brain = Brain(experience, salary)

for x in range(1000):
  brain.train()

guess = brain.predict(31.10)
print(guess, brain.compute_cost(guess))
brain.visualize()