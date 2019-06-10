import ablate
import numpy as np

p = ablate.Parameters()
p.add_parameters(
    x = {'type': 'variable'},
    y = {'type': 'variable'},
    c = {
        'value': 3.2,
        'type': 'constant',
    },
    t = {'type': 'dependence'},
    m = {'type': 'dependence'},
)

print(p)

p['t'] = np.linspace(0,1,num=4)
p['m'] = np.linspace(10,55,num=4)

print(p)

print(p['x'])