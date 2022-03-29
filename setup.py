from setuptools import setup

setup(
        name='CompositeView', 
        version='1.1.0', 
        description='A package for visualizing composite score data',
	author='Stephen Allegri',
	author_email='sallegri3@gatech.edu',
	packages=['composite_view'],
	install_requires=[
                'dash==2.0.0', 
                'dash_cytoscape==0.2.0', 
                'dash_bootstrap_components==1.0.1', 
                'numpy', 
                'pandas', 
                'networkx==2.6.3', 
                'colour', 
                'scikit-learn>=1.0.2', 
                'scipy>=1.7.1'
                ]
)