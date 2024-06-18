import gurobipy as gp
import pandas as pd
import numpy as np
import random
import math

random.seed(103093)

MAP_SIZE = 1000
NUMBER_OF_CUSTOMERS = 20
NUMBER_OF_POTENTIAL_LOCATION = NUMBER_OF_CUSTOMERS // 3
NUMBER_OF_EXISTING_COMPETITIVE_FACILITIES = NUMBER_OF_POTENTIAL_LOCATION // 3
NUMBER_OF_AVAILABLE_LOCATIONS = NUMBER_OF_POTENTIAL_LOCATION - NUMBER_OF_EXISTING_COMPETITIVE_FACILITIES
NUMBER_OF_NEW_FACILITIES = NUMBER_OF_AVAILABLE_LOCATIONS // 2

ALPHA = 0.05
BETA = 1
LAMBDA = 0.5
UPSILON = 3
ATTRACTIVENESS = 100
EPSILON = ALPHA


# Help functions
def get_utility(a, d):
    return a / (d ** BETA)


def get_h(customer):
    utility_sum = 0
    for e in E:
        utility_sum += get_utility(ATTRACTIVENESS, distances.get((customer, e)))
    return utility_sum


def get_u_bar(customer):
    utility_sum = 0
    for e in E_bar:
        utility_sum += get_utility(ATTRACTIVENESS, distances.get((customer, e)))
    return utility_sum


def get_interval_limit(customer):
    return get_u_bar(customer) + get_h(customer)


def get_g(utility):
    if utility == 0:
        return 0
    return 1 - math.exp(-LAMBDA * utility)


def get_g_derivative(utility):
    if utility == 0:
        return 0
    return LAMBDA * math.exp(-LAMBDA * utility)


def get_omega(utility, customer):
    if utility == 0:
        return 0
    return get_g(utility) * (1 - (get_h(customer) / utility))


def get_omega_derivative(utility, customer):
    if utility == 0:
        return 0
    return get_g_derivative(utility) * (1 - (get_h(customer) / utility)) + get_g(utility) * (
            get_h(customer) / (utility ** 2))


def get_l(utility, customer, c):
    return get_omega(c, customer) + get_omega_derivative(c, customer) * (utility - c)


def samesign(a, b):
    return a * b > 0


def diff_function_25(utility, customer, c):
    return get_l(utility, customer, c) - get_omega(utility, customer) * (1 + ALPHA)


def diff_function_24(utility, customer, c):
    return (get_omega_derivative(utility, customer) * (utility - c)
            + get_omega(c, customer) - get_omega(utility,customer))


def bisect24(low, high, customer, c):
    temp = high
    midpoint = (low + high) / 2.0
    while (high - low)/2 >= 0.001:
        midpoint = (low + high) / 2.0
        if samesign(diff_function_24(low, customer, c), diff_function_24(midpoint, customer, c)):
            low = midpoint
        else:
            high = midpoint
    if midpoint >= (1 - 0.001) * temp:
        return temp
    return midpoint

def bisect25(low, high, customer, c):
    temp = high
    midpoint = (low + high) / 2.0
    while (high - low)/2 >= 0.001:
        midpoint = (low + high) / 2.0
        if samesign(diff_function_25(low, customer, c), diff_function_25(midpoint, customer, c)):
            low = midpoint
        else:
            high = midpoint
    if midpoint >= (1 - 0.001) * temp:
        return temp
    return midpoint


# Data generation
N = [_ for _ in range(NUMBER_OF_CUSTOMERS)]
P = random.sample(N, NUMBER_OF_POTENTIAL_LOCATION)
E = random.sample(P, NUMBER_OF_EXISTING_COMPETITIVE_FACILITIES)
E_bar = [_ for _ in P if _ not in E]

N_ind = [_ for _ in range(NUMBER_OF_CUSTOMERS)]
P_ind = [_ for _ in range(NUMBER_OF_POTENTIAL_LOCATION)]
E_ind = [_ for _ in range(NUMBER_OF_EXISTING_COMPETITIVE_FACILITIES)]
E_bar_ind = [_ for _ in range(NUMBER_OF_AVAILABLE_LOCATIONS)]
S_ind = [_ for _ in range(NUMBER_OF_NEW_FACILITIES)]

# Create location value for node in N
locations = {}
for i in N_ind:
    x = random.randint(1, MAP_SIZE - 1)
    y = random.randint(1, MAP_SIZE - 1)

    if (x, y) not in locations.values():
        locations.update({i: (x, y)})
    else:
        while (x, y) in locations.values():
            x = random.randint(1, MAP_SIZE - 1)
            y = random.randint(1, MAP_SIZE - 1)
        locations.update({i: (x, y)})

# Create distance matrix
distances = {}
for i in N_ind:
    for j in N_ind:
        if i == j:
            distances.update({(i, j): 0.1})
        else:
            x1, y1 = locations.get(i)
            x2, y2 = locations.get(j)
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            distances.update({(i, j): dist})

# Create demand-weight
w = []
for i in N_ind:
    w.append(random.randint(5, 10))

# for i in N_ind:
#     print(get_interval_limit(i))

# The TLA procedure
l_dict = {}
a_dict = {}
b_dict = {}
c_dict = {}
for customer in N_ind:
    def tla():
        # Step 1
        l = 1
        c = get_h(customer)
        c_t = get_h(customer)
        b_dict.update({(customer, 1): get_omega_derivative(get_h(customer), customer)})

        c_dict.update({(customer, l): get_h(customer)})
        phi_bar = get_interval_limit(customer)

        # Step 2
        while get_l(phi_bar, customer, c) >= get_omega(phi_bar, customer) * (1 + ALPHA):
            print("Calculate root" + " - Customer " + str(customer) + " - c = " + str(c))
            root = bisect25(c, phi_bar, customer, c)
            c_dict.update({(customer, l + 1): root})
            if root == phi_bar:
                print("root = phi_bar" + " - Customer" + str(customer))
                l_dict.update({customer: l})
                break
            else:
                c = root
                print("Root:    " + str(root))
                print("Phi_bar: " + str(phi_bar))
                # Step 3
                l = l + 1
                if get_omega(phi_bar, customer) >= (
                        get_omega_derivative(phi_bar, customer) * (phi_bar - c) + get_omega(c, customer)):  # Step 3b
                    print("3b" + " - " + str(l) + " - Customer" + str(customer))
                    c_t = bisect24(c, phi_bar, customer, c)
                    b_dict.update({(customer, l): get_omega_derivative(c_t, customer)})
                else: # Step 3a
                    print("3a" + " - " + str(l) + " - Customer" + str(customer))
                    l_dict.update({customer: l})
                    c_dict.update({(customer, l): phi_bar})
                    if get_omega(c_dict.get((customer, l)), customer) * (1 + ALPHA) <= get_omega(phi_bar, customer):
                        value = (get_omega(phi_bar, customer) - get_omega(c_dict.get((customer, l)), customer) * (1 + ALPHA)) / (phi_bar - c)
                        b_dict.update({(customer, l): value})
                    else:
                        b_dict.update({(customer, l): 0})
                        break
                if c_t == phi_bar:
                    print("c_t == phi_bar" + " - Customer" + str(customer))
                    c_dict.update({(customer, l + 1): c_t})
                    l_dict.update({customer: l})
                    break

        if get_l(phi_bar, customer, c) < get_omega(phi_bar, customer) * (1 + ALPHA):
            l_dict.update({customer: l})
            c_dict.update({(customer, l + 1): phi_bar})


    tla()

for cus in N_ind:
    for l in range(1, l_dict.get(cus) + 1):
        a_dict.update({(cus, l) : c_dict.get((cus, l + 1)) - c_dict.get((cus, l)) })

print(l_dict)
print(c_dict)
print(b_dict)
print(a_dict)