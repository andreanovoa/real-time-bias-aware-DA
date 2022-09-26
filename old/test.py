class Ensemble2(type):
    name = 'Ensemble'

    def __new__(mcs, parent, attrdict, attrdict2):
        print(f'Creating a new {mcs.__name__} object...')
        name = parent.__name__ + 'Ensemble'
        # Create and initialise thermoacoustic class
        cls = type.__new__(mcs, 'name', (parent,), attrdict)
        cls.__init_subclass__(attrdict)

        print(cls.__class__)
        print(cls.__subclasses__())

        return cls

    def __init__(mcs, parent, attrdict, attrdict2):
        super().__init__(mcs)
        print(super())
        print('init')


    def toy(cls):
        print('toy func')



class Ensemble3:
    name = 'Ensemble'

    def __new__(mcs, name, parent, attrdict):
        # return parent(attrdict)
        print('__new__ called')
        print(parent.__name__)
        # obj = super().__new__(mcs, parent.__name__, (parent,), attrdict)
        # return obj

        # x = super().__new__(parent)
        x.__init__(attrdict)
        print(x.getParams())


        return x

    def __call__(self, *args, **kwargs):
        print('__call__ called')
        self.__init__(self, *args, **kwargs)

    def __init__(self, name, parent, attrdict):
        print('__init__ called')
        self.a=0




from Rijke2 import Case

module = Case
TA_p = {}
DA_p = dict(m=10)
# A = Ensemble3('Ensemble', module, TA_p)
A = Ensemble2(module, TA_p, DA_p)

print(A.__class__)
print(A.Nm)

print(A.timeIntegrate)
print(A.toy)

state, time = A.timeIntegrate(averaged=False)
A.updateHistory(state, time)
A.viewHistory()
