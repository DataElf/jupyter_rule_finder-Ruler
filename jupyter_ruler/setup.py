#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name='jupyter_rule_strategy',
    version='0.1.0',
    description='Interactive rule strategy development toolkit for Jupyter',
    packages=find_packages(),
    install_requires=[
        'ipywidgets>=7.6',
        'plotly>=4.0',
        'pandas>=1.0',
        'numpy',
        'scikit-learn>=0.24',
    ],
    python_requires='>=3.7',
)

