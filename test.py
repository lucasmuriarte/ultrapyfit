from ultrafast.utils.divers import LabBook, book_annotate

def bookmethod_registry():
    registry = {}

    def register(function):
        registry[function.__name__] = function
        return function

    register.track = registry
    return register

class ExperimentModule:
    def __init__(self, e):
        self.report = LabBook()
        self.e = e
        print(self.bookmethod.track)
        for method_name in self.bookmethod.track:
            setattr(self, method_name, book_annotate(self.report)(getattr(self, method_name)))

class Experiment:
    def __init__(self):
        self.preprocessing = self._Preprocessing(self)
        self.fit = self._Fit(self)
    

    class _Preprocessing(ExperimentModule):
        bookmethod = bookmethod_registry()

        @bookmethod
        def baseline_substraction(self, number_spec=2, only_one=False):
            pass
        
        @bookmethod
        def baseline_substraction2(self, number_spec=2, only_one=False):
            pass

    class _Fit(ExperimentModule):
        bookmethod = bookmethod_registry()

        @bookmethod
        def baseline_substraction(self, number_spec=2, only_one=False):
            pass
        
        @bookmethod
        def baseline_substraction2(self, number_spec=2, only_one=False):
            pass

e = Experiment()
e.preprocessing.baseline_substraction(['data'])
e.preprocessing.report.print()
e.fit.baseline_substraction2(['fit'])
e.fit.report.print()