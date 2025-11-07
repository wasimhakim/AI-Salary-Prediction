import matplotlib.pyplot as plt
import numpy as np

class Brain:
  w = 0
  b = 0
  learning_rate = 0.00001

  def __init__(self, experience, salary):
    self.experience = experience
    self.salary = salary
    self.data_size = len(self.experience)
  
  def predict(self, exp = None):
    x = exp if exp else self.experience
    return self.w * x + self.b

  def train(self):
    y_pred = self.predict()
    # error = self.compute_cost(y_pred)
    # gradient descent
    dj_dw = np.sum((y_pred - self.salary) * self.experience) / self.data_size
    dj_db = np.sum(y_pred - self.salary) / self.data_size
    self.w = self.w - self.learning_rate * dj_dw
    self.b = self.b - self.learning_rate * dj_db


  def compute_cost(self, y_pred):
    cost = np.sum((y_pred - self.salary) ** 2)
    total_cost = 1 / (2 * self.data_size) * cost
    return total_cost

  def visualize(self):
    plt.scatter(self.experience, self.salary) 
    plt.xlabel('Experience (in months)')
    plt.ylabel('Salary (in thousands)')
    plt.title('Experience vs Salary')
    plt.plot(self.experience, self.predict())
    plt.show()