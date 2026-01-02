import typing
import time
import pandas as pd
import numpy as np
import core.vars as vars
import core.base as base
import os

import matplotlib.pyplot as plt

import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

def naive_baies(texts: pd.Series, targets: pd.Series, delta: float=0.1) -> pd.Series:
    # Предобработка текста
    tokens, vocabulary = base.text_preprocessor(texts=texts, min_freq=1)

    voc_set = set(vocabulary.keys())

    # Априорная вероятность классов
    prior_probability = targets.value_counts(normalize=True)
    # Постериорная вероятность встретить слово в каждом из классов
    posterior_probability = pd.DataFrame(data=0, index=list(vocabulary.keys()), columns=prior_probability.index, dtype=int)
    # Регуляризация через delta
    posterior_probability = posterior_probability.map(lambda x: x + delta)

    for i in range(len(tokens)):
        document = tokens.iloc[i]
        target = targets.iloc[i]

        for word in document.split():
            idx = vocabulary.get(word, vocabulary.get(vars.UNK_VAL))
            posterior_probability.iloc[idx][target] += 1

    for k in posterior_probability.columns:
        k_sum = posterior_probability[k].sum()
        posterior_probability[k] = posterior_probability[k].apply(lambda x: x / k_sum)

    print(posterior_probability.iloc[list(range(50))])

    # Предсказываем класс
    pred_targets_values = []

    for i in range(len(tokens)):
        document = tokens.iloc[i]
        words = [word if word in voc_set else vars.UNK_VAL for word in document.split()]

        prediction = pd.Series(index=prior_probability.index, data=0.0)

        for k in prior_probability.index:
            # Берём априорную вероятность класса k
            k_probability = prior_probability.loc[k]
            # Берём постериорные вероятности слов words при условии класса k
            k_x_probability = posterior_probability.loc[words][k]
            # Перемножаем постериорные вероятности
            k_x_probability = k_x_probability.prod()

            # Перемножаем априорную и постериорные вероятности
            prediction[k] = k_probability * k_x_probability

        
        # Выбираем наиболее вероятный класс для данного текста
        pred_k = prior_probability.index[prediction.argmax()]

        print(words)
        print(prediction.sort_values(ascending=False))
        print(pred_k)

        pred_targets_values.append(pred_k)

    # Строим результирующее предсказание для выборки
    pred_targets = pd.Series(pred_targets_values)

    return pred_targets

class VanillaRNN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, alpha: float=1e-2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha

        # веса RNN
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros(hidden_size)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_grad(self, h):
        return 1.0 - h ** 2
    
    def forward(self, xs) -> np.ndarray:
        """
        xs: список входных векторов [(input_size,)]
        """
        h = np.zeros(self.hidden_size)

        self.cache = {
            "xs": [],
            "hs": [h],
        }

        for x in xs:
            z = self.Wx @ x + self.Wh @ h + self.bh
            h = self.tanh(z)

            self.cache["xs"].append(x)
            self.cache["hs"].append(h)

        return h
    
    def backward(self, dh):
        # инициализация градиентов
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        # BPTT
        for t in reversed(range(len(self.cache["xs"]))):
            h_t = self.cache["hs"][t + 1]
            h_prev = self.cache["hs"][t]
            x_t = self.cache["xs"][t]

            dz = dh * self.tanh_grad(h_t)

            dWx += np.outer(dz, x_t)
            dWh += np.outer(dz, h_prev)
            dbh += dz

            dh = self.Wh.T @ dz

        # обновление весов
        self.Wx -= self.alpha * dWx
        self.Wh -= self.alpha * dWh
        self.bh -= self.alpha * dbh
    
class BiRNNClassifier:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, alpha: float=1e-2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.cache = {}

        # веса выхода
        self.Wy = np.random.randn(output_size, 2*hidden_size) * 0.01
        self.by = np.zeros(output_size)

    def softmax(self, x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)

    def output_repr(self, h_T):
        # выход только по последнему состоянию
        logits = self.Wy @ h_T + self.by
        probs = self.softmax(logits)
        self.cache["probs"] = probs

        return logits
    
    def probs(self, logits: np.ndarray=None):
        if logits:
            return self.softmax(logits)
        else:
            return self.cache["probs"]

    def loss(self, probs, y_true):
        """
        y_true: int — индекс правильного класса
        """
        return -np.log(probs[y_true] + 1e-9)
    
    def backward(self, y_true, h_T):
        # инициализация градиентов
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)

        # градиент выхода
        probs = self.cache["probs"]
        dlogits = probs.copy()
        dlogits[y_true] -= 1  # формула градиента для выхода

        dWy += np.outer(dlogits, h_T)
        dby += dlogits

        dh = self.Wy.T @ dlogits

        # обновление весов
        self.Wy -= self.alpha * dWy
        self.by -= self.alpha * dby

        return dh[:self.hidden_size], dh[self.hidden_size:]

def check_naive_baies(texts: pd.Series, targets: pd.Series):
    pred_targets = naive_baies(texts=texts, targets=targets)

    print(pd.concat([targets, pred_targets], axis=1))

    # Сверяем эталон с нашими предсказаниями
    print(accuracy_score(targets, y_pred=pred_targets))

def get_vocabulary(by: str="download") -> tuple[typing.Any, int]:
    if by == "download":
        model = api.load('glove-twitter-100', return_path=False)
        return model, model.vector_size
    elif by == "preload":
        model_path = api.load('glove-twitter-100', return_path=True)

        model = KeyedVectors.load_word2vec_format(
            model_path,
            binary=False,
            # no_header=True
        )
        return model, model.vector_size
    else:
        return None, None
    
def get_vec_tokens(
        texts: list[list[str]], 
        vocabulary: pd.Series, 
        vector_size: int
    ):
    tokens = []

    print("tokens")

    UNK = np.zeros(vector_size)
    vocabulary_tokens = []
    vocabulary_vectors = []

    for i in range(len(texts)):
        words = texts.iloc[i].split()
        xs = []
        for word in words:
            if word in vocabulary:
                xs.append(vocabulary[word])
                vocabulary_tokens.append(word)
                vocabulary_vectors.append(vocabulary[word])
            else:
                xs.append(UNK)
        tokens.append(xs)

    return tokens

def bi_train_step(
        rnn_f: VanillaRNN,
        rnn_b: VanillaRNN,
        rnn_c: BiRNNClassifier,
        xs: list,
        y_true_idx: int,
    ):
    h_T_forward = rnn_f.forward(xs)
    h_T_backward = rnn_b.forward(xs[::-1])
    h_cat = np.concatenate([h_T_forward, h_T_backward])

    logits = rnn_c.output_repr(h_cat)
    probs = rnn_c.probs()
    loss = rnn_c.loss(probs, y_true_idx)

    dh_f, dh_b = rnn_c.backward(y_true_idx, h_cat)
    rnn_f.backward(dh_f)
    rnn_b.backward(dh_b)

    return loss

def show_roc_curve(y_true_all, y_score_all, roc_auc):
    fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"BiRNN (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for BiRNN")
    plt.legend()
    plt.grid(True)
    plt.show()

def check_vanilla_rnn(texts: pd.Series, targets: pd.Series):
    print("VanillaRNN")

    vocabulary, vector_size = get_vocabulary(by="preload")

    # Настраиваем гиперпараметры
    input_size = vector_size
    hidden_size = 20
    output_size = targets.nunique()
    n_epoches = 50
    alpha = 0.003

    print("text_preprocessor")

    # Разбиваем выборку на тренировочную и проверочную
    train_texts, test_texts, train_targets, test_targets = train_test_split(texts, targets, test_size=0.2)

    # Предобработка текста
    train_texts_cleared, _ = base.text_preprocessor(texts=train_texts, min_freq=1)
    test_texts_cleared, _ = base.text_preprocessor(texts=test_texts, min_freq=1)

    # Получаем ембеддинги для слов
    train_tokens = get_vec_tokens(
        texts=train_texts_cleared,
        vocabulary=vocabulary,
        vector_size=vector_size
    )
    test_tokens = get_vec_tokens(
        texts=test_texts_cleared,
        vocabulary=vocabulary,
        vector_size=vector_size
    )

    # Инициализируем 2 RNN - одна для промого обхода, другая для обратного
    rnn_forward = VanillaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        alpha=alpha
    )
    rnn_backward = VanillaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        alpha=alpha
    )
    # Инициализируем классификатор
    rnn_classifier = BiRNNClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        alpha=alpha
    )

    # Создаем набор возможных классов (всего 2 класса)
    classes = pd.Series(data=targets.unique())

    print("epoches")

    start_train_time = time.time()

    losses = []

    # Начинаем обучение
    for epoch in range(n_epoches):
        epoch_losses = [] 
        for i in range(len(train_texts_cleared)):
            y_true = train_targets.iloc[i]
            y_true_idx = classes[classes == y_true].index
            xs = train_tokens[i]

            # Делаем шаг обучения
            loss = bi_train_step(
                rnn_f=rnn_forward,
                rnn_b=rnn_backward,
                rnn_c=rnn_classifier,
                xs=xs,
                y_true_idx=y_true_idx
            )

            # Логируем лосс для визуальной проверки в конце - что RNN действительно обучается
            epoch_losses.append(np.round(loss, 5))

        epoch_losses_series = pd.Series(epoch_losses, name=f"epoch_{epoch+1}")

        print(f"epoch: {epoch}")
        losses.append(epoch_losses_series)

    # Смотрим лосс
    print(f"Seconds trained: {time.time() - start_train_time}")
    losses_df = pd.concat(losses, axis=1)
    print(losses_df)

    # Проверяем на тестовой выборке + выводим график ROC-AUC
    y_pred = []
    y_true_all = []
    y_score_all = []

    print("y_pred")

    for i in range(len(test_texts_cleared)):
        xs = test_tokens[i]

        h_T_forward = rnn_forward.forward(xs)
        h_T_backward = rnn_backward.forward(xs[::-1])
        h_cat = np.concatenate([h_T_forward, h_T_backward])

        logits = rnn_classifier.output_repr(h_cat)

        probs = pd.Series(rnn_classifier.probs())

        y_true_idx = probs.argmax()
        y_pred.append(classes[y_true_idx])

        y_true_all.append(test_targets.iloc[i])
        y_score_all.append(probs[1])   # вероятность класса 1

    pred_targets = pd.Series(y_pred)

    # Вывод метрики точности
    print(pd.concat([test_targets, pred_targets], axis=1))
    print(accuracy_score(test_targets, y_pred=pred_targets))

    # Вывод ROC-AUC
    roc_auc = roc_auc_score(y_true_all, y_score_all)
    print(f"ROC-AUC (BiRNN): {roc_auc:.4f}")

    # Визуализация ROC кривой
    show_roc_curve(y_true_all, y_score_all, roc_auc)

if __name__ == "__main__":
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    data = pd.read_csv(os.path.join(vars.DATASETS_ROOT, "comments.tsv"), sep='\t')

    texts = data['comment_text']
    targets = data['should_ban']
    
    check_vanilla_rnn(texts, targets)

    