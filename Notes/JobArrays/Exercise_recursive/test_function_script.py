import random as rnd
import math as math


def run_random_func(run_time):
    test_list = []
    sum_test = 0
    for k in range(run_time):
        for i in range(run_time):
            rnd.seed(0)
            test_list.append(rnd.random())
            print("Random number in list is", test_list[i])
            for j in test_list:
                print("Number in test list is", j)
                sum_test += math.sqrt(j)
                print("sum is", sum_test)
        test_list = []
        print("\n")
        print("End of iteration", k)
        print("\n")


if __name__ == '__main__':
    # it will run for ~ 1 min.
    run_random_func(5)
