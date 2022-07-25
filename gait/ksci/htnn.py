import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from NN_REG import Data_generator as gen

plt.style.use("seaborn-whitegrid")

X, sigma_square, epsilon, y = gen.DataGenerator(num_of_data=1000).get()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)

plt.plot(X, y, ".")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Total dataset")
plt.show()


def plot_results(x, y, y_est_mu, title, y_est_std=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, ".", label="y")
    plt.plot(x, y_est_mu, "-y", label="y_estimate_mu")
    plt.plot(x, np.sin(4 * x) * np.sin(5 * x), "-g", label="true_mu")
    if y_est_std is not None:
        plt.plot(x, y_est_mu + 2 * y_est_std, "-r", label="estimate_mu+2std")
        plt.plot(x, y_est_mu - 2 * y_est_std, "-r", label="estimate_mu-2std")

    plt.legend()
    plt.title(title)
    plt.show()


def plot_model_results(model, x, y, title, tfp_model: bool = True):
    si = np.argsort(x)
    x = x[si]
    y = y[si]
    yhat = model(x)
    if tfp_model:
        y_est_mu = yhat.mean()
        y_est_std = yhat.stddev()
    else:
        y_est_mu = yhat
        y_est_std = None
    plot_results(x, y, y_est_mu, title, y_est_std)


def negloglik(y, distr):
    return -distr.log_prob(y)


model_lin_reg_tfp = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(100, activation="sigmoid"),
        tf.keras.layers.Dense(1),
    ]
)

model_lin_reg_tfp.compile(
    optimizer=tf.optimizers.SGD(learning_rate=0.05), loss=tf.keras.losses.mse
)
model_lin_reg_tfp.summary()
history = model_lin_reg_tfp.fit(x_train, y_train, epochs=6000, verbose=0)

plot_model_results(model_lin_reg_tfp, x_train, y_train, 'Standard NN model train result', tfp_model=False)
plot_model_results(model_lin_reg_tfp, x_test, y_test, 'Standard NN model test result', tfp_model=False)

model_lin_reg_std_nn_tfp = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(100, activation="sigmoid"),
        tf.keras.layers.Dense(2),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(
                loc=t[:, 0:1], scale=tf.math.softplus(t[:, 1:2])
            )
        ),
    ]
)

model_lin_reg_std_nn_tfp.compile(
    optimizer=tf.optimizers.SGD(learning_rate=0.05), loss=negloglik
)
model_lin_reg_std_nn_tfp.summary()
history = model_lin_reg_std_nn_tfp.fit(x_train, y_train, epochs=6000, verbose=0)

plot_model_results(model_lin_reg_std_nn_tfp, x_train, y_train, 'Heteroscedastic NN model train result', tfp_model=True)
plot_model_results(model_lin_reg_std_nn_tfp, x_test, y_test, 'Heteroscedastic NN model test result', tfp_model=True)
results = pd.DataFrame(index=["Train", "Test"])

models = {
    "Linear regression": model_lin_reg_tfp,
    "Neural network + std": model_lin_reg_std_nn_tfp,
}
rmse = tf.keras.metrics.RootMeanSquaredError()
for model in models:
    results[model] = [
        rmse(y_train[:, tf.newaxis], models[model](x_train)
             ).numpy(),
        rmse(y_test[:, tf.newaxis], models[model](x_test)
             ).numpy(),
    ]
results.transpose()
print(results)

results_2 = pd.DataFrame(index=["Train", "Test"])

models = {
    "Linear regression": model_lin_reg_tfp,
    "Neural network + std": model_lin_reg_std_nn_tfp,
}
mae = tf.keras.metrics.MeanAbsoluteError()
for model in models:
    results_2[model] = [
        mae(y_train[:, tf.newaxis], models[model](x_train))
            .numpy(),
        mae(y_test[:, tf.newaxis], models[model](x_test))
            .numpy(),
    ]
results_2.transpose()
print(results_2)