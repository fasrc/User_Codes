## Purpose:

Build a mamba environment with `pandas` and `numpy`, as part of a
batch job, and use that environment to execute a Python script.

## Contents:

1. `build_env.sh`: Bash script to build mamba environment

2. `numpy_pandas_ex.py`: Python source code

3. `run.sbatch`: Batch-job submission script for sending the job to the queue

## Example Usage:

**Step 1:** Build mamba environment as part of your batch job.

Contents of `build_env.sh`:

```bash
mamba create -n my_env python pip wheel pandas numpy -y
```

Include the following, in your batch-job submission script, to build the
environment:

```bash
sh build_env.sh
```

**Step 2:** Run `numpy_pandas_ex.py` by submitting batch job

Contents of `numpy_pandas_ex.py`:

```python
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
```

Submit batch job:

```bash
sbatch run.sbatch
```

### Example Output:

Content of output file `np_pandas.out` showing *only* the newly
created dataframe:

```bash
        0      1       2
0   Spark  20000  30days
1  pandas  20000  40days
  Courses    Fee Duration
a   Spark  20000   30days
b  pandas  20000   40days
    Courses      Fee Duration  Discount
r0    Spark  22000.0    30day      1000
r1  PySpark  25000.0   50days      2300
r2   Hadoop  23000.0   55days      1000
r3   Python  24000.0   40days      1200
r4   Pandas      NaN   60days      2500
r5     None  25000.0    35day      1300
r6    Spark  25000.0               1400
r7   Python  22000.0   50days      1600
```
