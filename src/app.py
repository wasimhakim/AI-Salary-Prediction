import pandas as pd
from brain import Brain

salary_data = pd.read_csv('./data/Experience-Salary.csv')

brain = Brain(salary_data)
m = 100
for x in range(m):
  brain.train()

while True:
  print("1. Predict - 2. Visualize")
  choice = int(input("Enter your choice: "))
  match choice:
    case 1:
      experience = float(input("Enter experience: "))
      guess = brain.predict(experience)
      print("Salary: ", guess)
    case 2:
      brain.visualize()
    case _:
      break;