import matplotlib.pyplot as plt

class Brain:
  w = 0
  b = 0
  y_pred = 0

  def __init__(self, experience, salary):
    self.experience = experience
    self.salary = salary
  
  def train(self):
    self.y_pred = self.w * self.experience + self.b

  def visualize(self):
    plt.scatter(self.experience, self.salary) 
    plt.xlabel('Experience (in months)')
    plt.ylabel('Salary (in thousands)')
    plt.title('Experience vs Salary')
    plt.plot(self.experience, self.y_pred)
    plt.show()