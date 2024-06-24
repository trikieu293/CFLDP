import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
import itertools

random.seed(123)

def cfldp(n_customer, alpha, beta, lamda, theta):
    MAP_SIZE = 1000
    CUSTOMERS = n_customer
    BUDGET = 20
    ATTRACTIVENESS_ATTRIBUTES = 2

    POTENTIAL_LOCATION = CUSTOMERS // 3
    EXISTING_COMPETITIVE_FACILITIES = POTENTIAL_LOCATION // 3
    AVAILABLE_LOCATIONS = POTENTIAL_LOCATION - EXISTING_COMPETITIVE_FACILITIES


    ALPHA = alpha            # approximation level
    BETA = beta              # the distance sensitivity parameter
    LAMBDA = lamda           # the elasticity parameter
    THETA = theta            # sensitivity parameter of the utility function

    N = [node for node in range(1, CUSTOMERS + 1)]          # index of customers
    P = random.sample(N, POTENTIAL_LOCATION)                # index of potential locations
    C = random.sample(P, EXISTING_COMPETITIVE_FACILITIES)   # index of competitive facility locations
    S = [facility for facility in P if facility not in C]   # index of controlled facilities

    # initiating locations for nodes
    locations = {}
    for i in N:
        x = random.randint(1, MAP_SIZE - 1)
        y = random.randint(1, MAP_SIZE - 1)

        while (x, y) in locations.values():
            x = random.randint(1, MAP_SIZE - 1)
            y = random.randint(1, MAP_SIZE - 1)
        locations.update({i: (x, y)})
    print("Locations: "+str(locations))

    # calculating distances between nodes
    distances = {}
    for i in N:
        for j in N:
            if i == j:
                distances.update({(i, j): 0.1})
            else:
                x1, y1 = locations.get(i)
                x2, y2 = locations.get(j)
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distances.update({(i, j): distance})
    print("Distances: "+str(distances))

    # creating dict of scenarios
    scenarios_attr = {}
    for k in range(1, ATTRACTIVENESS_ATTRIBUTES + 1):
        value = 3
        scenarios_attr.update({k: [level for level in range(value)]})

    product = itertools.product(*[k for k in scenarios_attr.values()])
    R = {}
    count = 1
    for item in product:
        R.update({count: list(item)})
        count += 1

    # creating dict of nodes' weight
    w = {}
    for i in N:
        w.update({i: random.randint(1, 5)})
    # initiating attractiveness for competitive facilities
    C_attractiveness = {}
    for c in C:
        C_attractiveness.update({c: random.randint(1, 10)})

    ### Help functions
    def get_attractiveness(scenario):
        # return 1 + 1 * sum(R.get(scenario))
        attractiveness = 1
        for i in R.get(scenario):
            attractiveness = attractiveness*((1 + i)**THETA)
        return attractiveness
    def get_cost(scenario):
        return 1 + 0.5 * sum(R.get(scenario))

    def get_utility(customer, facility, scenario):
        return get_attractiveness(scenario) * (distances.get((customer, facility)) + 1) ** (-BETA)

    def get_g(utility):
        if utility == 0:
            return 0
        return 1 - math.exp(-LAMBDA * utility)

    def get_g_derivative(utility):
        if utility == 0:
            return 0
        return LAMBDA * math.exp(-LAMBDA * utility)

    def get_u_c(customer):
        utility_sum = 0
        for c in C:
            utility_sum += C_attractiveness.get(c) * (distances.get((customer, c)) + 1) ** (-BETA)
        return utility_sum

    def get_max_u_s(customer):
        utility_sum = 0
        for e in S:
            utility_sum += get_utility(customer, e, max(R.keys()))
        return utility_sum

    def get_interval_limit(customer):
        return get_u_c(customer) + get_max_u_s(customer)

    def get_omega(utility, customer):
        if utility == 0:
            return 0
        return get_g(utility) * (1 - (get_u_c(customer) / utility))

    def get_omega_derivative(utility, customer):
        if utility == 0:
            return 0
        return (get_g_derivative(utility) * (1 - (get_u_c(customer) / utility))
                + get_g(utility) * (get_u_c(customer) / (utility ** 2)))

    def get_l(utility, customer, point):
        return get_omega(point, customer) + get_omega_derivative(point, customer) * (utility - point)

    def is_same_sign(a, b):
        return a * b > 0

    def diff_function_25(utility, customer, point):
        return get_l(utility, customer, point) - get_omega(utility, customer) * (1 + ALPHA)


    def diff_function_24(utility, customer, point):
        return (get_omega_derivative(utility, customer) * (utility - point)
                + get_omega(point, customer) - get_omega(utility, customer))

    def bisect(func, low, high, customer, c):
        temp = high
        midpoint = (low + high) / 2.0
        while (high - low)/2 >= 0.001:
            midpoint = (low + high) / 2.0
            if is_same_sign(func(low, customer, c), func(midpoint, customer, c)):
                low = midpoint
            else:
                high = midpoint
        if midpoint >= (1 - 0.0001) * temp:
            return temp
        return midpoint

    def data_callback(model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            gap = (abs(cur_bd - cur_obj) / abs(cur_obj)) * 100

            # Change in obj value or bound?
            if model._obj != cur_obj or model._bd != cur_bd:
                model._obj = cur_obj
                model._bd = cur_bd
                model._gap = gap
                model._data.append([time.time() - model._start, cur_obj, cur_bd, gap])

    ### The TLA procedure
    l_dict = {}
    a_dict = {}
    b_dict = {}
    c_dict = {}
    for customer in N:
        def tla():
            # Step 1
            l = 1
            c = get_u_c(customer)
            c_t = get_u_c(customer)
            b_dict.update({(customer, 1): get_omega_derivative(get_u_c(customer), customer)})

            c_dict.update({(customer, l): get_u_c(customer)})
            phi_bar = get_interval_limit(customer)

            # Step 2
            while get_l(phi_bar, customer, c) >= get_omega(phi_bar, customer) * (1 + ALPHA):
                # print("Calculate root" + " - Customer " + str(customer) + " - c = " + str(c))
                root = bisect(diff_function_25, c, phi_bar, customer, c)
                c_dict.update({(customer, l + 1): root})
                if root == phi_bar:
                    # print("root = phi_bar" + " - Customer" + str(customer))
                    l_dict.update({customer: l})
                    break
                else:
                    c = root
                    l = l + 1
                    if get_omega(phi_bar, customer) >= (
                            get_omega_derivative(phi_bar, customer) * (phi_bar - c) + get_omega(c, customer)):  # Step 3b
                        c_t = bisect(diff_function_24, c, phi_bar, customer, c)
                        b_dict.update({(customer, l): get_omega_derivative(c_t, customer)})
                    else: # Step 3a
                        l_dict.update({customer: l})
                        c_dict.update({(customer, l): phi_bar})
                        if get_omega(c_dict.get((customer, l)), customer) * (1 + ALPHA) <= get_omega(phi_bar, customer):
                            value = (get_omega(phi_bar, customer) - get_omega(c_dict.get((customer, l)), customer) * (1 + ALPHA)) / (phi_bar - c)
                            b_dict.update({(customer, l): value})
                        else:
                            b_dict.update({(customer, l): 0})
                            break
                    if c_t == phi_bar:
                        c_dict.update({(customer, l + 1): c_t})
                        l_dict.update({customer: l})
                        break

            if get_l(phi_bar, customer, c) < get_omega(phi_bar, customer) * (1 + ALPHA):
                l_dict.update({customer: l})
                c_dict.update({(customer, l + 1): phi_bar})
        tla()

    for cus in N:
        for l in range(1, l_dict.get(cus) + 1):
            a_dict.update({(cus, l) : c_dict.get((cus, l + 1)) - c_dict.get((cus, l))})

    ### Model
    model = gp.Model()
    x_index = [(j, r) for j in S for r in R.keys()]

    # Decision varibales
    x = model.addVars(x_index, vtype=GRB.INTEGER, name='x')
    y = model.addVars([(i, l) for (i, l) in b_dict.keys()], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y')

    # Objective Function
    model.setObjective(sum(sum(w[i] * a_dict.get((i, l)) * b_dict.get((i, l)) * y[i, l] for l in range(1, l_dict.get(i) + 1)) for i in N), GRB.MAXIMIZE)

    # Constraints
    for i in N:
        model.addConstr(sum(get_utility(i, j, r) * x[j, r] for j in S for r in R.keys()) == sum(a_dict.get((i, l)) * y[i, l] for l in range(1, l_dict.get(i) + 1)), name="Constraints 1")

    for j in S:
        model.addConstr(sum(x[j, r] for r in R.keys()) <= 1, name="Constraints 2")

    model.addConstr(sum(sum(get_cost(r) * x[j, r] for r in R.keys()) for j in S) <= (AVAILABLE_LOCATIONS // 2) * get_cost(max(R.keys())), name="Constraints 3")

    model._obj = None
    model._bd = None
    model._gap = None
    model._data = []
    model._start = time.time()
    model.Params.TimeLimit = 60*60
    model.update()

    # model.optimize(callback=data_callback)
    model.optimize()

    ### checking result
    x_result = pd.DataFrame(x.keys(), columns=["j", "r"])
    x_result["value"] = model.getAttr("X", x).values()
    x_result.drop(x_result[x_result.value < 0.9].index, inplace=True)
    x_result["attractiveness"] = [get_attractiveness(r) for r in x_result["r"]]

    y_result = pd.DataFrame(y.keys(), columns=["i", "l"])
    y_result["value"] = model.getAttr("X", y).values()

    return [x_result, y_result, a_dict, b_dict, c_dict, l_dict]


if __name__ == "__main__":
    result = cfldp(40, 0.001, 1, 0.5, 0.9)