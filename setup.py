from setuptools import setup

setup(
        name='visualizer', 
        version='1.1.0', 
        description='A package for visualizing SemNet results',
	author='Stephen Allegri',
	author_email='sallegri3@gatech.edu',
	packages=['visualizer'],
	install_requires=[
                'dash==2.0.0', 
                'dash_cytoscape==0.2.0', 
                'dash_bootstrap_components==1.0.1', 
                'numpy', 
                'pandas', 
                'networkx==2.6.3', 
                'colour'
                ]
)