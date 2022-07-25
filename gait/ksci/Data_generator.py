import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, num_of_data=1000):
        np.random.seed(1234)
        self.x = (np.pi / 2)*np.random.random_sample(num_of_data)
        self.target_error_var = 0.02 + 0.02 * (1-np.sin(4*self.x))**2
        self.target_error = np.zeros_like(self.target_error_var)
        for idx, error in enumerate(self.target_error_var):
            self.target_error[idx] = np.random.normal(0, np.sqrt(error), 1)
        self.target_error_ = np.random.normal(0, np.sqrt(self.target_error_var), num_of_data)
        # print(self.target_error)
        # print(self.target_error_)
        self.target = np.sin(4*self.x)*np.sin(5*self.x) + self.target_error
        # print(self.target_error_var)

    def get(self):
        return self.x, self.target_error_var, self.target_error, self.target

    def plot(self):

        plt.scatter(self.x, self.target)
        plt.show()