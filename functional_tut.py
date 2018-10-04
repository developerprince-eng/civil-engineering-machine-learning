from __future__ import print_function
def apply_twice(func, arg):
    return func(func(arg))
def sum_of_three(x):
    return (x + 3)

print(apply_twice(sum_of_three, 10))