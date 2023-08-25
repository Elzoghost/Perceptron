import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PerceptronClassifier:
    def __init__(self, eta0=0.1, max_iter=100, random_state=1):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state = random_state
        self.ppn = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_std = None
        self.X_test_std = None
        self.pca = None
        self.X_train_pca = None
        self.X_test_pca = None
    
    def load_data(self):
        # Charger les données
        iris = load_iris()
        X = iris.data[:, [2, 3]]
        y = iris.target

        # Diviser les données en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    def standardize_data(self):
        # Standardiser les caractéristiques
        sc = StandardScaler()
        self.X_train_std = sc.fit_transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

    def reduce_dimension(self):
        # Réduire la dimensionnalité avec PCA
        self.pca = PCA(n_components=2)
        self.X_train_pca = self.pca.fit_transform(self.X_train_std)
        self.X_test_pca = self.pca.transform(self.X_test_std)

    def train(self):
        # Entraîner le modèle avec la version de Perceptron de scikit-learn
        self.ppn = Perceptron(eta0=self.eta0, max_iter=self.max_iter, random_state=self.random_state)
        self.ppn.fit(self.X_train_pca, self.y_train)

    def predict(self):
        # Faire des prédictions
        y_pred = self.ppn.predict(self.X_test_pca)
        print('Accuracy: %.2f' % np.mean(y_pred == self.y_test))

    def plot_performance(self):
        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(range(1, len(self.ppn.errors_) + 1), self.ppn.errors_, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Number of updates')
        ax.set_title('Perceptron')
        plt.show()


if __name__ == '__main__':
    pc = PerceptronClassifier()
    pc.load_data()
    pc.standardize_data()
    pc.reduce_dimension()
    pc.train()
    pc.predict()
    #pc.plot_performance()
