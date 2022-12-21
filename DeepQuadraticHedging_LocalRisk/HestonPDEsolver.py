'''

This script adapts the code from

https://github.com/redbzi/NM-Heston

for solving the Heston PDE arising from the local-risk minimization
problem in one dimension, starting from the modified Heston model in

[Cerný, A., & Kallsen, J. (2008)]

'''

import numpy as np
from math import sinh, asinh
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from math import exp


'''

Coefficients for the discretization

'''

def delta_s(i, pos, Delta_s):
    if pos == -1:
        return 2 / (Delta_s[i] * (Delta_s[i] + Delta_s[i + 1]))
    elif pos == 0:
        return -2 / (Delta_s[i] * Delta_s[i + 1])
    elif pos == 1:
        return 2 / (Delta_s[i + 1] * (Delta_s[i] + Delta_s[i + 1]))
    else:
        raise ValueError("Wrong pos")


def delta_v(i, pos, Delta_v):
    if pos == -1:
        return 2 / (Delta_v[i] * (Delta_v[i] + Delta_v[i + 1]))
    elif pos == 0:
        return -2 / (Delta_v[i] * Delta_v[i + 1])
    elif pos == 1:
        return 2 / (Delta_v[i + 1] * (Delta_v[i] + Delta_v[i + 1]))
    else:
        raise ValueError("Wrong pos")


# def alpha_s(i, pos, Delta_s):
#     if pos == -2:
#         return Delta_s[i] / (Delta_s[i - 1] * (Delta_s[i - 1] + Delta_s[i]))
#     elif pos == -1:
#         return (-Delta_s[i - 1] - Delta_s[i]) / (Delta_s[i - 1] * Delta_s[i])
#     elif pos == 0:
#         return (Delta_s[i - 1] + 2 * Delta_s[i]) / (Delta_s[i] * (Delta_s[i - 1] + Delta_s[i]))
#     else:
#         raise ValueError("Wrong pos")


def alpha_v(i, pos, Delta_v):
    if pos == -2:
        return Delta_v[i] / (Delta_v[i - 1] * (Delta_v[i - 1] + Delta_v[i]))
    elif pos == -1:
        return (-Delta_v[i - 1] - Delta_v[i]) / (Delta_v[i - 1] * Delta_v[i])
    elif pos == 0:
        return (Delta_v[i - 1] + 2 * Delta_v[i]) / (Delta_v[i] * (Delta_v[i - 1] + Delta_v[i]))
    else:
        raise ValueError("Wrong pos")


def beta_s(i, pos, Delta_s):
    if pos == -1:
        return -Delta_s[i + 1] / (Delta_s[i] * (Delta_s[i] + Delta_s[i + 1]))
    elif pos == 0:
        return (Delta_s[i + 1] - Delta_s[i]) / (Delta_s[i] * Delta_s[i + 1])
    elif pos == 1:
        return Delta_s[i] / (Delta_s[i + 1] * (Delta_s[i] + Delta_s[i + 1]))
    else:
        raise ValueError("Wrong pos")


def beta_v(i, pos, Delta_v):
    if pos == -1:
        return -Delta_v[i + 1] / (Delta_v[i] * (Delta_v[i] + Delta_v[i + 1]))
    elif pos == 0:
        return (Delta_v[i + 1] - Delta_v[i]) / (Delta_v[i] * Delta_v[i + 1])
    elif pos == 1:
        return Delta_v[i] / (Delta_v[i + 1] * (Delta_v[i] + Delta_v[i + 1]))
    else:
        raise ValueError("Wrong pos")


# def gamma_s(i, pos, Delta_s):
#     if pos == 0:
#         return (-2 * Delta_s[i + 1] - Delta_s[i + 2]) / (Delta_s[i + 1] * (Delta_s[i + 1] + Delta_s[i + 2]))
#     elif pos == 1:
#         return (Delta_s[i + 1] + Delta_s[i + 2]) / (Delta_s[i + 1] * Delta_s[i + 2])
#     elif pos == 2:
#         return -Delta_s[i + 1] / (Delta_s[i + 2] * (Delta_s[i + 1] + Delta_s[i + 2]))
#     else:
#         raise ValueError("Wrong pos")


def gamma_v(i, pos, Delta_v):
    if pos == 0:
        return (-2 * Delta_v[i + 1] - Delta_v[i + 2]) / (Delta_v[i + 1] * (Delta_v[i + 1] + Delta_v[i + 2]))
    elif pos == 1:
        return (Delta_v[i + 1] + Delta_v[i + 2]) / (Delta_v[i + 1] * Delta_v[i + 2])
    elif pos == 2:
        return -Delta_v[i + 1] / (Delta_v[i + 2] * (Delta_v[i + 1] + Delta_v[i + 2]))
    else:
        raise ValueError("Wrong pos")

'''

Sample grid

'''

def Map_s(xi, K, c):
    return K + c * sinh(xi)


def Map_v(xi, d):
    return d * sinh(xi)


def make_grid(m1, S, S_0, K, c, m2, V, V_0, d):
    Delta_xi = (1.0 / m1) * (asinh((S - K) / c) - asinh(-K / c))
    Uniform_s = [asinh(-K / c) + i * Delta_xi for i in range(m1 + 1)]
    Vec_s = [Map_s(Uniform_s[i], K, c) for i in range(m1 + 1)]
    Vec_s.append(S_0)
    Vec_s.sort()
    Vec_s.pop(-1)
    Delta_s = [Vec_s[i + 1] - Vec_s[i] for i in range(m1)]
    
    Delta_eta = (1.0 / m2) * asinh(V / d)
    Uniform_v = [i * Delta_eta for i in range(m2 + 1)]
    Vec_v = [Map_v(Uniform_v[i], d) for i in range(m2 + 1)]
    Vec_v.append(V_0)
    Vec_v.sort()
    Vec_v.pop(-1)
    Delta_v = [Vec_v[i + 1] - Vec_v[i] for i in range(m2)]

    X, Y = np.meshgrid(Vec_s, Vec_v)

    # # grid checking
    # plt.plot(X, Y, '.', color='blue')
    # plt.show()

    return Vec_s, Delta_s, Vec_v, Delta_v, X, Y

'''

PDE solver

'''


def F(n, omega, A, b, r_f, delta_t):
    return A * omega + b * exp(r_f * delta_t * n)


def F_0(n, omega, A_0, b_0, r_f, delta_t):
    return A_0 * omega + b_0 * exp(r_f * delta_t * n)


def F_1(n, omega, A_1, b_1, r_f, delta_t):
    return A_1 * omega + b_1 * exp(r_f * delta_t * n)


def F_2(n, omega, A_2, b_2, r_f, delta_t):
    return A_2 * omega + b_2 * exp(r_f * delta_t * n)


def MCS_scheme(m, N, U_0, delta_t, eta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    start = datetime.now()
    U = np.zeros([m, N+1])
    U[:, 0] = U_0
    I = np.identity(m)
    lhs_1 = csc_matrix(I - eta * delta_t * A_1)
    inv_lhs_1 = inv(lhs_1)
    lhs_2 = csc_matrix(I - eta * delta_t * A_2)
    inv_lhs_2 = inv(lhs_2)
    
    for n in range(0, N):
        Y_0 = U[:, n] + delta_t * F(n - 1, U[:, n], A, b, r_f, delta_t)
        rhs_1 = Y_0 + eta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U[:, n], A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + eta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U[:, n], A_2, b_2, r_f, delta_t))  #we update b_2
        Y_2 = inv_lhs_2 * rhs_2
        Y_0_hat = Y_0 + eta * delta_t * (F_0(n, Y_2, A_0, b_0, r_f, delta_t) - F_0(n - 1, U[:, n], A_0, b_0, r_f, delta_t))
        Y_0_tilde = Y_0_hat + (0.5 - eta) * delta_t * (F(n, Y_2, A, b, r_f, delta_t) - F(n - 1, U[:, n], A, b, r_f, delta_t))

        rhs_1 = Y_0_tilde + eta * delta_t * (b_1 * exp(r_f * delta_t * n) - F_1(n - 1, U[:, n], A_1, b_1, r_f, delta_t))  #we update b_1
        Y_1_tilde = inv_lhs_1 * rhs_1
        
        rhs_2 = Y_1_tilde + eta * delta_t * (b_2 * exp(r_f * delta_t * n) - F_2(n - 1, U[:, n], A_2, b_2, r_f, delta_t))  #we update b_2
        U[:, n+1] = inv_lhs_2 * rhs_2
 
    end = datetime.now()
    time = (end - start).total_seconds()
    return U, time


'''

Matrices factory for the splitting method

'''

def make_matrices(m1, m2, m, mu,  rho, sigma, r_d, r_f, kappa, theta, Vec_s, Vec_v, Delta_s, Delta_v):
    A_0 = np.zeros((m, m))
    A_1 = np.zeros((m, m))
    A_2 = np.zeros((m, m))

    l_9a = [-2, -1, 0]
    l_9b = [-1, 0, 1]
    l_9c = [0, 1, 2]
    l_10 = [-1, 0, 1]
    l_11 = [[-1, 0, 1], [-1, 0, 1]]

    # Definition of A_0
    for j in range(1, m2):
        for i in range(1, m1):
            c = rho * sigma * Vec_s[i] * Vec_v[j]
            for k in l_11[0]:
                for l in l_11[1]:
                    A_0[i + j * (m1 + 1), (i + k) + (j + l) * (m1 + 1)] += c * beta_s(i - 1, k, Delta_s) * beta_v(j - 1, l, Delta_v)

    A_0 = csc_matrix(A_0)
    #plt.spy(A_0)
    #plt.show()

    # Definition of A_1
    for j in range(m2 + 1):
        for i in range(1, m1):
            a = 0.5 * Vec_s[i] ** 2 * Vec_v[j]
            b = (r_d - r_f) * Vec_s[i]
            for k in l_10:
                A_1[i + j * (m1 + 1), (i + k) + j * (m1 + 1)] += (a * delta_s(i - 1, k, Delta_s) + b * beta_s(i - 1, k, Delta_s))
            A_1[i + j * (m1 + 1), i + j * (m1 + 1)] += - 0.5 * r_d
        A_1[m1 + j * (m1 + 1), m1 + j * (m1 + 1)] += - 0.5 * r_d


    A_1 = csc_matrix(A_1)
    #plt.spy(A_1)
    #plt.show()

    #Definition of A_2
    for j in range(m2 - 1):
        for i in range(m1 + 1):
            temp = (kappa * theta + rho * sigma * r_d ) - (kappa + rho * sigma * mu ) *  Vec_v[j]
            temp2 = 0.5 * sigma ** 2 * Vec_v[j]
            if Vec_v[j] > 1.:
                for k in l_9a:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp * alpha_v(j, k, Delta_v)
                for k in l_10:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp2 * delta_v(j - 1, k, Delta_v)
            if j == 0:
                for k in l_9c:
                    A_2[i, i + (m1 + 1) * k] += temp * gamma_v(j, k, Delta_v)
            else:
                for k in l_10:
                    A_2[i + j * (m1 + 1), i + (m1 + 1) * (j + k)] += (temp * beta_v(j - 1, k, Delta_v) + temp2 * delta_v(j - 1, k, Delta_v))
            A_2[i + j * (m1 + 1), i + j * (m1 + 1)] += - 0.5 * r_d

    A_2 = csc_matrix(A_2)
    #plt.spy(A_2)
    #plt.show()

    A = A_0 + A_1 + A_2
    A = csc_matrix(A)
    #plt.spy(A)
    #plt.show()

    return A_0, A_1, A_2, A


def make_boundaries(m1, m2, m, r_d, r_f, N, Vec_s, delta_t):
    b_0 = [0.] * m
    b_1 = [0.] * m
    b_2 = [0.] * m

    # Boundary when s = S
    for j in range(m2 + 1):
        b_1[m1 * (j + 1)] = (r_d - r_f) * Vec_s[-1] * exp(-r_f * delta_t * (N - 1))

    # Boundary when v = V
    for i in range(1, m1 + 1):
        b_2[m - m1 - 1 + i] = -0.5 * r_d * Vec_s[i] * exp(-r_f * delta_t * (N - 1))

    b_0 = np.array(b_0)
    b_1 = np.array(b_1)
    b_2 = np.array(b_2)

    b = b_0 + b_1 + b_2

    return b_0, b_1, b_2, b


'''

Interpolation functions

'''

def get_price(Vec_s, Vec_v, s, v, price_grid):
 
    i, j = 0, 0
    while Vec_s[i] < s and i < len(Vec_s)-1: i += 1
    while Vec_v[j] < v and j < len(Vec_v)-1: j += 1
    
    if j > 0:
         x1, x2 = Vec_s[(i-1):(i+1)]
         y1, y2 = Vec_v[(j-1):(j+1)]
         q11, q12 = price_grid[j-1, (i-1):(i+1)]
         q21, q22 = price_grid[j, (i-1):(i+1)]

         # Bilinear interpolation    
         return np.array([[x2-s, s-x1]]).dot(np.array([[q11, q12], [q21, q22]])).dot(np.array([[y2-v], [v-y1]])/((x2-x1)*(y2-y1)))
    
    else:
         x1, x2 = Vec_s[(i-1):(i+1)]
         q21, q22 = price_grid[0, (i-1):(i+1)]
         # linear interpolation
         return (s-x2)*q21/(x1-x2)-(s-x1)*q22/(x1-x2)
     
        
def interpolate_price(Vec_s, Vec_v, price_grid, num_strajectories, num_time_interval, XY):        

    price_pde = np.zeros((num_strajectories, num_time_interval+1))   

    for m in range(num_strajectories):
        for n in range(num_time_interval+1):
            ss = XY[m, 0, n]
            vv = np.maximum(XY[m, 1, n], 0)
            price_pde[m, n] = get_price(Vec_s, Vec_v, ss, vv, price_grid[:, :, n])

    return price_pde
 
    
def make_derivative_matrices(m1, m2, m, Vec_s, Vec_v, Delta_s, Delta_v):
    A_1 = np.zeros((m, m)) # matrix for the derivative w.r.t. s
    A_2 = np.zeros((m, m)) # matrix for the derivative w.r.t. v

    l_9a = [-2, -1, 0]
    l_9c = [0, 1, 2]
    l_10 = [-1, 0, 1]
    
    # Definition of A_1
    for j in range(m2 + 1):
        for i in range(1, m1):
            for k in l_10:
                A_1[i + j * (m1 + 1), (i + k) + j * (m1 + 1)] +=  beta_s(i - 1, k, Delta_s)

    A_1 = csc_matrix(A_1)

    #Definition of A_2
    for j in range(m2 - 1):
        for i in range(m1 + 1):
            if Vec_v[j] > 1.:
                for k in l_9a:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += alpha_v(j, k, Delta_v)
            if j == 0:
                for k in l_9c:
                    A_2[i, i + (m1 + 1) * k] += gamma_v(j, k, Delta_v)
            else:
                for k in l_10:
                    A_2[i + j * (m1 + 1), i + (m1 + 1) * (j + k)] += beta_v(j - 1, k, Delta_v)

    A_2 = csc_matrix(A_2)

    return A_1, A_2
