# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from deploy import app

#%% Launch the FlaskPy dev server
if __name__ == '__main__':
    app.run(host="localhost", debug=True)