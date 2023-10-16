# example from https://sparkbyexamples.com/pandas/pandas-dataframe-tutorial-beginners-guide/

import numpy as np
import pandas as pd

# Create pandas DataFrame from List
import pandas as pd
technologies = [ ["Spark",20000, "30days"], 
                 ["pandas",20000, "40days"], 
               ]
df=pd.DataFrame(technologies)
print(df)

# Add Column & Row Labels to the DataFrame
column_names=["Courses","Fee","Duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=column_names,index=row_label)
print(df)

# set custom types to DataFrame
types={'Courses': str,'Fee':float,'Duration':str}
df=df.astype(types)

# Create DataFrame with None/Null to work with examples
technologies   = ({
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas",None,"Spark","Python"],
    'Fee' :[22000,25000,23000,24000,np.nan,25000,25000,22000],
    'Duration':['30day','50days','55days','40days','60days','35day','','50days'],
    'Discount':[1000,2300,1000,1200,2500,1300,1400,1600]
          })
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df = pd.DataFrame(technologies, index=row_labels)
print(df)

df.describe()

