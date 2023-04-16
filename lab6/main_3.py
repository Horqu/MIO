import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skfuzzy import control as ctrl

data = np.loadtxt('dane_pogodowe.csv', dtype='str', delimiter=',', skiprows=1)[:, 1:]
data = np.array(data, dtype='float')

temperature = ctrl.Antecedent(np.arange(-10, 30, 0.1),'temperature')
humidity = ctrl.Antecedent(np.arange(27, 83, 1),'humidity')
wind = ctrl.Antecedent(np.arange(0.5, 5.5, 0.05),'wind')
rain = ctrl.Antecedent(np.arange(0, 95, 0.01),'rain')
pm10 = ctrl.Consequent(np.arange(1, 115, 0.1),'pm10')
temp_ratio = 0.1
temperature['very cold'] = fuzz.trapmf(temperature.universe,[-10,-10,-6,0]) * temp_ratio
temperature['cold'] = fuzz.trapmf(temperature.universe,[-2,2,6,10]) * temp_ratio
temperature['average'] = fuzz.trapmf(temperature.universe,[8,11,15,18]) * temp_ratio
temperature['warm'] = fuzz.trapmf(temperature.universe,[17,19,23,25]) * temp_ratio
temperature['hot'] = fuzz.trapmf(temperature.universe,[24,26,30,30]) * temp_ratio
hum_ratio = 0.1
humidity['medium'] = fuzz.trapmf(humidity.universe,[27,27,35,45]) * hum_ratio
humidity['wet'] = fuzz.trapmf(humidity.universe,[40,48,57,67]) * hum_ratio
humidity['very wet'] = fuzz.trapmf(humidity.universe,[64,70,83,83]) * hum_ratio
wind_ratio = 1.0
wind['light'] = fuzz.trapmf(wind.universe,[0.5, 0.5, 1.5, 2.3]) * wind_ratio
wind['medium'] = fuzz.trapmf(wind.universe,[2.0, 2.7, 3.3, 3.9]) * wind_ratio
wind['strong'] = fuzz.trapmf(wind.universe,[3.7, 4.4, 5.5, 5.5]) * wind_ratio
rain_ratio = 1.0
rain['none'] = fuzz.trapmf(rain.universe,[0.0, 0.0, 0.0, 0.01]) * rain_ratio
rain['very light'] = fuzz.trapmf(rain.universe,[0.01, 0.5, 0.8, 1.0]) * rain_ratio
rain['light'] = fuzz.trapmf(rain.universe,[0.85, 1.2, 4.5, 5.0]) * rain_ratio
rain['average'] = fuzz.trapmf(rain.universe,[4.7, 5.5, 12.0, 20.0]) * rain_ratio
rain['strong'] = fuzz.trapmf(rain.universe,[16.0, 35.0, 95.0, 95.0]) * rain_ratio
pm10['very low'] = fuzz.trapmf(pm10.universe,[1,1,7,10])
pm10['low'] = fuzz.trapmf(pm10.universe,[8,15,27,35])
pm10['medium'] = fuzz.trapmf(pm10.universe,[30,45,75,90])
pm10['high'] = fuzz.trapmf(pm10.universe,[80,100,115,115])
#zasady
rule1 = ctrl.Rule(temperature['warm'] | temperature['hot'], pm10['very low'])
rule2 = ctrl.Rule(humidity['very wet'], pm10['very low'])
rule3 = ctrl.Rule(wind['strong'], pm10['very low'])
rule4 = ctrl.Rule(rain['average'] | rain['strong'], pm10['very low'])
rule5 = ctrl.Rule(temperature['cold'] | temperature['average'] | temperature['warm'], pm10['low'])
rule6 = ctrl.Rule(humidity['medium'] | humidity['wet'], pm10['low'])
rule7 = ctrl.Rule(wind['medium'] | wind['strong'], pm10['low'])
rule8 = ctrl.Rule(rain['average'] | rain['light'], pm10['low'])
rule9 = ctrl.Rule(temperature['cold'] | temperature['average'], pm10['medium'])
rule10 = ctrl.Rule(humidity['medium'], pm10['medium'])
rule11 = ctrl.Rule(wind['light'], pm10['medium'])
rule12 = ctrl.Rule(rain['none'] | rain['very light'], pm10['medium'])
rule13 = ctrl.Rule(temperature['very cold'], pm10['high'])
rule14 = ctrl.Rule(wind['light'], pm10['high'])
rule15 = ctrl.Rule(rain['none'], pm10['high'])
#model
control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
model = ctrl.ControlSystemSimulation(control_system)
model.input['temperature'] = 0.43
model.input['humidity'] = 57
model.input['wind'] = 2.19
model.input['rain'] = 0.0
model.compute()
print('temp: 0.43 humidity: 57 wind: 2.19 rain: 0, real pm10: 2.18')
print('calculated pm10:', model.output['pm10'])