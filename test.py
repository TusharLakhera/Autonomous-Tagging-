# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:14:50 2018

@author: asus
"""

import datetime
import pandas as pd


Questions=pd.read_csv('Questions.csv',encoding='latin-1')

def Dateconverter(creationdate):
    z=creationdate[0:10]
    znew=datetime.datetime.strptime(z,'%Y-%m-%d')
    week=datetime.datetime.strftime(znew,'%W')
    return week
Questions['week']=Questions['CreationDate']
Questions.week=Questions.week.apply(Dateconverter)
Questions.week[0]