##The following makes use of pandas to display population of diffrenet cities and the amount of rainfall each city experiences

##from __future__ import print_function 
##Uncomment the line above if you running your code in the early versions of python like python 2.7 

import pandas as pd
import	numpy as np


city_names = pd.Series(['Bulwayo', 'Harare', 'Kwekwe', 'Gweru', 'Mutare'])
population = pd.Series([7888778, 2445456, 2776868, 4455544, 5546466])
rainfall = pd.Series([180, 70, 67, 90, 120])
pp_log = np.log(population)

pp = pd.DataFrame({'City Name': city_names, 'Population': population, 'Ranifall(mm)': rainfall, 'Log of Population': pp_log})

pp.index
print("")
print(pp.index)
print(type(pp[0:1]))
pp[0:1]
print("")

print(pp)
print("")
