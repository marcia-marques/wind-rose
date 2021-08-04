# wind-rose

Python library to plot wind roses.

## Installation

You can install directly from this repository:

```
pip install git+https://github.com/marcia-marques/wind-rose.git
```

## Example of use

```python
import windroses.windroses as wr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('data.txt')

plt.figure(figsize=(9,9))
sns.set_style('darkgrid')
wr.wind_rose(df, 'wd', yaxis=67.5)
```
![image](https://user-images.githubusercontent.com/75334161/128109752-82a96fd0-0380-42c8-8542-c301ba34e4a5.png)
