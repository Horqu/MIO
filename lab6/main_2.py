import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(15, 35, 0.1), 'temperature')

humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'humidity')

sun = ctrl.Antecedent(np.arange(0, 100, 1), 'sun')

watering = ctrl.Consequent(np.arange(0, 25, 0.1), 'watering')

temperature['chlodno'] = fuzz.gaussmf(temperature.universe, 15, 2.5)
temperature['przecietnie'] = fuzz.gaussmf(temperature.universe, 20, 2.5)
temperature['cieplo'] = fuzz.gaussmf(temperature.universe, 25, 2.5)
temperature['goraco'] = fuzz.gaussmf(temperature.universe, 30, 2.5)
temperature['bgoraco'] = fuzz.gaussmf(temperature.universe, 35, 2.5)

humidity['bsucho'] = fuzz.gaussmf(humidity.universe, 0, 12.5)
humidity['sucho'] = fuzz.gaussmf(humidity.universe, 25, 12.5)
humidity['przecietnie'] = fuzz.gaussmf(humidity.universe, 50, 12.5)
humidity['mokro'] = fuzz.gaussmf(humidity.universe, 75, 12.5)
humidity['bmokro'] = fuzz.gaussmf(humidity.universe, 100, 12.5)

watering['nie podlewaj'] = fuzz.gaussmf(watering.universe, 0, 2.5)
watering['malo podlewaj'] = fuzz.gaussmf(watering.universe, 5, 2.5)
watering['srednio podlewaj'] = fuzz.gaussmf(watering.universe, 10, 3.5)
watering['duzo podlewaj'] = fuzz.gaussmf(watering.universe, 18, 3.75)
watering['maksymalnie podlewaj'] = fuzz.gaussmf(watering.universe, 25, 4)

sun['brak naslonecznienia'] = fuzz.gaussmf(sun.universe, 0, 12.5) * 0.8
sun['pochmurno'] = fuzz.gaussmf(sun.universe, 25, 12.5) * 0.8
sun['male naslonecznienie'] = fuzz.gaussmf(sun.universe, 50, 12.5) * 0.8
sun['srednie naslonecznienie'] = fuzz.gaussmf(sun.universe, 75, 12.5) * 0.8
sun['mocne naslonecznienie'] = fuzz.gaussmf(sun.universe, 100, 12.5) * 0.8
# temperature.view()
# humidity.view()
# watering.view()
# sun.view()

rule1 = ctrl.Rule(
    ( humidity['bmokro'] & (temperature['chlodno'] | temperature['przecietnie'] | temperature['cieplo'])) | 
    ( humidity['mokro'] & temperature['chlodno']) 
    , watering['nie podlewaj'])

rule2 = ctrl.Rule(
    ( humidity['przecietnie'] & temperature['chlodno'] ) | 
    ( humidity['mokro'] & (temperature['przecietnie'] | temperature['cieplo'] | temperature['goraco'])) | 
    ( humidity['bmokro'] & temperature['bgoraco'])
    , watering['malo podlewaj'])

rule3 = ctrl.Rule(
    ( humidity['sucho'] & (temperature['chlodno'] | temperature['przecietnie'])) | 
    ( humidity['przecietnie'] & (temperature['przecietnie'] | temperature['cieplo'])) | 
    ( humidity['mokro'] & temperature['bgoraco'])
    , watering['srednio podlewaj'])

rule4 = ctrl.Rule(
    ( humidity['bsucho'] & (temperature['chlodno'] | temperature['przecietnie'] | temperature['cieplo'])) |
    ( humidity['sucho'] & (temperature['cieplo'] | temperature['goraco'])) |
    ( humidity['przecietnie'] & (temperature['goraco'] | temperature['bgoraco']))
    , watering['duzo podlewaj']
)

rule5 = ctrl.Rule(
    ( humidity['bsucho'] & (temperature['goraco'] | temperature['bgoraco'])) |
    ( humidity['sucho'] & (temperature['bgoraco']))
    , watering['maksymalnie podlewaj']
)

rule6 = ctrl.Rule(
    sun['brak naslonecznienia']
    , watering['nie podlewaj']
)
rule7 = ctrl.Rule(
    sun['pochmurno']
    , watering['malo podlewaj']
)
rule8 = ctrl.Rule(
    sun['male naslonecznienie']
    , watering['srednio podlewaj']
)
rule9 = ctrl.Rule(
    sun['srednie naslonecznienie']
    , watering['duzo podlewaj']
)
rule10 = ctrl.Rule(
    sun['mocne naslonecznienie']
    , watering['maksymalnie podlewaj']
)
control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
model = ctrl.ControlSystemSimulation(control_system)

sun_value = 100

model.input['temperature'] = 20
model.input['humidity'] = 50
model.input['sun'] = sun_value

model.compute()
print('Sun value in %: ' + str(sun_value))
print(model.output['watering'], ' liters/day')
watering.view(sim=model)

import seaborn as sns
import pandas as pd

temperature_grid, humidity_grid = np.meshgrid(np.arange(15, 35, 0.5), np.arange(0, 100, 1))
test_points = np.transpose(np.vstack((np.ravel(temperature_grid), np.ravel(humidity_grid))))

model.input['temperature'] = test_points[:,0]
model.input['humidity'] = test_points[:,1]
model.compute()

test_points = np.concatenate((test_points, model.output['watering'].reshape(-1,1)), axis=1)

sns.heatmap(pd.DataFrame(test_points, columns = ['temperature','humidity','watering']).pivot(index='humidity', columns='temperature', values='watering'), cmap = 'coolwarm')

input('Press Enter to continue...')