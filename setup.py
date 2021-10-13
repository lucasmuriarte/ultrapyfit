import setuptools

setuptools.setup(
	name="ultrafast",
	version='0.1.0',
	description='Ultrafast spectroscopy data treatment',
	url='https://luclabarriere.github.io/ultrafast/', #TODO :Change repo
	author='Lucas Martinez Uriarte',
	python_requires='>=3.6',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Framework :: Matplotlib',
		'Intended Audience :: Education',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Natural Language :: English',
		'Operating System :: MacOS',
		'Operating System :: Microsoft',
		'Operating System :: Unix',
		'Programming Language :: Python :: 3',
		'Topic :: Scientific/Engineering :: Chemistry',
		'Topic :: Scientific/Engineering :: Physics'
	],
	install_requires=[
		'matplotlib',
		'numpy',
		'pyqt5',
		'lmfit',
		'nose-parameterized',
		'pandas',
		'seaborn'
	]
)
