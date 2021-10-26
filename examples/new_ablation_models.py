'''Test atmosphere model registration implemenation
'''

import ablate

class A(ablate.AblationModel):
    ATMOSPHERES = {}

    def run(self, *args, **kwargs):
        pass


class B(ablate.AblationModel):
    def run(self, *args, **kwargs):
        pass


A._register_atmosphere('my_atm', lambda x: None, {'my_meta':None})

print(A.ATMOSPHERES)
print(ablate.AblationModel.ATMOSPHERES)


B._register_atmosphere('my_atm2', lambda x: None, {'my_meta11':None})

print(B.ATMOSPHERES)
print(ablate.AblationModel.ATMOSPHERES)