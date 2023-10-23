import numpy as np

if __name__ == '__main__':

    class Example3:

        defaults = dict(N_units=100,
                        caca=5
                        )

        __slots__ = list(defaults.keys()) + ["slot_0", "W"]

        def __init__(self, **kwargs):
            for key, value in Example3.defaults.items():
                if key in kwargs.keys():
                    value = kwargs[key]
                setattr(self, key, value)

        @property
        def N(self):
            return 'we have M'


    a = Example3()
    b = Example3(**{'caca':2})


    a.W = 22
    print(a.W)
    a.W = 0
    print(a.N)

    print(getattr(a, 'caca'))
    print(getattr(b, 'caca'))