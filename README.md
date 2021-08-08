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

df = pd.read_csv('data/pinheiros.txt')

sns.set_style('darkgrid')

plt.figure(figsize=(6,6))
wr.wind_rose(df, 'wd', yaxis=67.5)
```
![image](https://user-images.githubusercontent.com/75334161/128647815-4cdcee31-6d9e-485f-9478-a3285ca7ccb7.png)

```python
plt.figure(figsize=(9,9))
wr.wind_rose_season(df, 'wd', yaxis=67.5, yticks=[100, 300, 500, 700])
```
![image](https://user-images.githubusercontent.com/75334161/128647850-50fd6993-3456-4766-a4a3-6bba45ac1cb1.png)

```python
plt.figure(figsize=(6,6))
wr.wind_rose_speed(df, 'ws', 'wd', yaxis=67.5)
```
![image](https://user-images.githubusercontent.com/75334161/128647860-8e51cff6-181d-4f46-8e90-c23d531d8ebc.png)

```python
plt.figure(figsize=(9,9))
yticks = [200, 400, 600, 800]
lims = [0., 1.42, 2.84, 4.26, 5.68, 7.1 ]
wr.wind_rose_speed_season(df, 'ws', 'wd', yticks=yticks, lims=lims, yaxis=67.5)
```
![image](https://user-images.githubusercontent.com/75334161/128647875-756d3092-7b86-42cd-8cee-4791293ae777.png)

```python
plt.figure(figsize=(7, 7))
wr.wind_rose_pollution(df, 'co', 'ws', 'wd', var_label='CO (ppm)', yaxis=245)
```
![image](https://user-images.githubusercontent.com/75334161/128647882-40a0d9c5-8cec-4ff0-9456-18aaad8125f8.png)

```python
plt.figure(figsize=(10, 10))
lims = [0., 1.42, 2.84, 4.26, 5.68]
wr.wind_rose_pollution_season(df, 'co', 'ws', 'wd', var_label='CO (ppm)', lims=lims, yaxis=245)
```
![image](https://user-images.githubusercontent.com/75334161/128647887-74dbc4b6-d211-47ff-b37c-96c6a9924432.png)

## Acknowledgments

Air quality data provided by [CETESB](https://qualar.cetesb.sp.gov.br/qualar/home.do).

## License

The source code is released under the [BSD-3-Clause License](https://github.com/marcia-marques/wind-rose/blob/master/LICENSE).
