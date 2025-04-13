import numpy as np
import pickle
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    return accuracy

## 定义函数，导入训练好的模型，输出在测试集上的分类准确率（Accuracy）

def load_model_and_evaluate(model_path, X_test, y_test):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy