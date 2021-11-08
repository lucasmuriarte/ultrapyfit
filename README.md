# ultrafast
---
## Getting-started



# Choosing a hierarchy
The file names should be chosen to be:
	- small
	- self-explanatory
	- all-lowercase
For example, the matplotlib package as the following modules:
```python
matplotlib.mlab
matplotlib.offsetbox
matplotlib.patches
matplotlib.path
matplotlib.patheffects
...
```
For example, we could obtain this kind of structure:
```python
ultrafast.globalfit
ultrafast.globalfitbootstrap
ultrafast.globalparams
ultrafast.modelcreator
ultrafast.exponentialfit
ultrafast.targetfit
ultrafast.experiment
ultrafast.graphics.exploredata
ultrafast.graphics.exploreresults
ultrafast.graphics.plotsvd
ultrafast.graphics.cursors
ultrafast.graphics.targetmodel
ultrafast.chirpcorrection
ultrafast.misc
ultrafast.preprocessing

```
We should  also check wether some classes could be grouped together in one file, such as `GlobalFitExponential` and `GlobalFitTargetModel`
# Refactoring tests
We should never have lines of code lying outside the TestXXX class. For example:
```python
time_simulated = np.logspace(0, 3, 150)

path = 'examples/data/denoised_2.csv'
original_taus = [8, 30, 200]

class TestGlobalFit(unittest.TestCase):
    ...
```

A better way of doing this is to use the `setUpClass @classmethod` inside the TestXXX class:
```python
    @classmethod
    def setUpClass(self):
		self.time_simulated = np.logspace(0, 3, 150)
		
		self.path = 'examples/data/denoised_2.csv'
		self.original_taus = [8, 30, 200]
```
(Notice the use of `self.`)