# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:03:47 2022

@author: an553
"""

import os as os
import numpy as np


data = np.load(os.getcwd()+'\\data\\bias_VdP.npz')

exec(open("Bias.py").read(), {'filename':'bias_VdP', 
                              't_train': 1.0, 
                              't_wash': 0.05, 
                              'upsample': 5, 
                              'test_run': True})