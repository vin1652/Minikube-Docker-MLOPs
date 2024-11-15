import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class IrisModelTrainer:
    def __init__(self):
        self.model = None

    def load_data(self):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_and_save_model(self):
        X_train, X_test, y_train, y_test = self.load_data()
        self.model = LogisticRegression(random_state=42, solver='liblinear')
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"F1 Score: {f1}")
        
        with open('iris_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    trainer = IrisModelTrainer()
    trainer.train_and_save_model()