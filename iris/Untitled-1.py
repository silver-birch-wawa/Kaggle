import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
iris=pd.read_csv('iris.csv')
#print(iris)
sns.lmplot('PetalLengthCm','SepalLengthCm',iris,hue='Species', fit_reg=False)
plt.show()