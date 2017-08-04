def read_csv_into_df(csv_file, fields, cols):
    # import sys
    # sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages')
    # sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg')
    # sys.path.insert(0,'/vol/medic02/users/ag6516/miniconda/envs/PyTorch27/lib/python2.7/site-packages/IPython/extensions')
    # sys.path
    import pandas as pd

    df = pd.read_csv(csv_file, names=fields, usecols=cols, skiprows=1)

    # Remove rows with nan for 'Report text'
    df = df[pd.notnull(df['Report text'])]

    # Remove rows with clinical history in 'Report text' field
    for i, row in df.iterrows():
        if 'Clinical History' in row['Report text']:
            # print row['Report text']
            df.drop(i, inplace=True)

    # Drop duplicates
    df = df.drop_duplicates(subset=['Accession'])

    return df

