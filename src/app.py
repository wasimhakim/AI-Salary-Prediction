import pandas as pd
from brain import Brain

salary_data = pd.read_csv('./data/Experience-Salary.csv')

brain = Brain(salary_data)

for x in range(1000):
  brain.train()

guess = brain.predict(31.10)
print(guess, brain.compute_cost(guess))
brain.visualize()