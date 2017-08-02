import sys
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg')
sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/IPython/extensions')
sys.path

import csv
import string
import pandas as pd


fields = ['Accession', 'Clinical history', 'Comment', 'Report text']
dir = ('/vol/medic02/users/ag6516/x_ray_fracture_localisation/')

with open(dir + 'data/Reports/XKNEB_Jan_Jul_2017_anon_edited.csv', 'rU') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    headers = data.next()
    for row in data:
        if len(row) != len(headers):
            raise('Error')

print headers

cols = [headers.index(item) for item in fields]
print cols
#df = pd.read_csv
#df_clean = df.drop_duplicates(subset=['Accession'])

#df.to_csv('cleaned_reports.csv', sep=',')