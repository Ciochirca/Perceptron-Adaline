import numpy as np
import pandas as pd

import neurons
import plot

df = pd.read_csv('iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values #what is that 4
y = np.where(y == 'Iris-versicolor', -1, 1) #what are -1 and 1

X = df.iloc[0:100, [0, 2]].values

plot.plotSetosaVersicolor(X)

ppn = neurons.Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

plot.plotEpochUpdates(ppn)

plot.plotDecisionRegions(X, y, ppn)

ada = neurons.AdalineGD(eta=0.1, n_iter=10)
ada.fit(X, y)
plot.plotDecisionRegions(X, y, ada)