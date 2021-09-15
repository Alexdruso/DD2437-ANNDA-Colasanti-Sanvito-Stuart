from data import *
from two_layer_perceptron import TwoLayerPerceptron

data = generate_binary_classification_data()
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

p = TwoLayerPerceptron(
    mode='online',
    learning_rate=1e-3,
    momentum=0.9,
    max_iterations=300,
    tolerance=None,
    hidden_layer_size=10,
    validation_fraction=0.2,
)

p.fit(X, y)
pred = p.predict(X, y)
print('Mean Square Error: {}'.format(p.loss_))
