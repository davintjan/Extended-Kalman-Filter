import numpy as np
import matplotlib.pyplot as plt

N = 1000
# Ground truth or Expected Value
a_truth = -1
expected_value = np.full(N,a_truth)

# Distribution parameters for noise
mean_eps, sigma_eps = 0, 1  # System noise
mean_nu, sigma_nu = 0, 0.5  # Observation noise

# Distribution parameters for initial x0
mean_x0, sigma_x0 = 1, 2

# Generate noise
epsilon = np.random.normal(mean_eps, sigma_eps, size=N)
nu = np.random.normal(mean_nu, sigma_nu, size=N)

def measurement_function(x):
    return np.sqrt(x ** 2 + 1)

# Initialize observation
x = np.zeros(N+1)
x[0] = np.random.normal(mean_x0, sigma_x0)
y = np.zeros(N)

for i in range(N):
    y[i] = measurement_function(x[i]) + nu[i]
    x[i + 1] = a_truth * x[i] + epsilon[i]

# Time to implement EKF
a_guess = np.zeros(N+1)
a_guess[0] = -5
Mux = np.zeros(N+1)
Mux[0] = x[0]

current_cov = np.array([[4, 0], [0, 0.00001]])

def compute_measurement_jacobian(x):
    return np.array([[x / np.sqrt(x ** 2 + 1), 0]])

def EKF(current_estimate, current_cov, a_guess, epsilon, observation):
    Q = 0.5
    R = np.array([[1, 0], [0, 0.000001]])

    # Propagation
    new_Mx = a_guess* current_estimate + epsilon
    x_a = np.array([[new_Mx], [a_guess]]) # 2X1
    A = np.array([[a_guess , current_estimate], [0, 1]]) #2X2
    new_cov = A@current_cov@A.T + R

    # Update step
    C = compute_measurement_jacobian(new_Mx)
    # C = np.array([[measurement_function(new_Mx), 0]]) #Check, may be wrong
    K = new_cov @ C.T / (C @ new_cov @ C.T + Q)
    print(K)
    Mx_update = x_a + K * (observation - measurement_function(new_Mx))
    cov_update = (np.eye(2) - K @ C) @ new_cov

    return Mx_update[0,0], Mx_update[1,0], cov_update

# A + sigma


for i in range(N):

    Mx, Ma, cov = EKF(Mux[i], current_cov, a_guess[i], epsilon[i], y[i])
    Mux[i+1] = Mx
    a_guess[i+1] = Ma
    current_cov = cov


a_pos = a_guess + 2
a_neg = a_guess -2

# Plotting
#
# plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
# plt.plot(y, label='Observations')
# plt.plot(expected_value, label='Expected A Value')
# plt.xlabel('Time Step')
# plt.ylabel('State Value')
# plt.title('Ground Truth vs Estimated States')
# plt.legend()
#
#
# plt.show()

plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
plt.plot(a_guess, label='Predicted_A')
plt.plot(a_pos, label='Ma + sigma')
plt.plot(a_neg, label='Ma - sigma')
plt.plot(expected_value, label='Expected_value')
plt.xlabel('Time Step')
plt.ylabel('State Value')
plt.title('Ground Truth vs Estimated States')
plt.legend()


plt.show()

