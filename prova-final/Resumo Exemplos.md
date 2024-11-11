
# Processo de Modelagem de Machine Learning

## Nível 2: Escolha de Modelo

### Divisão de Dados: Train, Validation e Test

Ao construir um modelo de machine learning, é essencial dividir o conjunto de dados para avaliar a performance e evitar overfitting. Existem três conjuntos principais:

- **Train (Treino)**: Usado para treinar o modelo.
- **Validation (Validação)**: Usado para ajustar hiperparâmetros e evitar overfitting.
- **Test (Teste)**: Usado para avaliar a performance final do modelo.

### Exemplo em Python de Train-Validation-Test Split

```python
from sklearn.model_selection import train_test_split

# Carregar os dados (exemplo)
# Suponha que temos `X` para features e `y` para o alvo
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

### Cross-Validation e Train-Test Split

Outra técnica é a validação cruzada (cross-validation), em que o conjunto de treino é dividido em várias partes, treinando em algumas e validando em outras. Isso permite uma melhor avaliação da generalização do modelo.

- **Train-Test Split**: Dividir o dataset em treino e teste.
- **Cross-Validation**: Divide o conjunto de treino em várias "dobras" (folds), garantindo uma avaliação mais robusta.

#### Exemplo de Cross-Validation em Python

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Modelo exemplo
model = LogisticRegression()

# Validação cruzada com 5 dobras
scores = cross_val_score(model, X, y, cv=5)
print("Acurácias por dobra:", scores)
print("Acurácia média:", scores.mean())
```

## Nível 1: Certificação

Certifique-se de que o modelo está pronto para produção e atende aos critérios de desempenho esperados. Esse passo inclui verificar métricas de performance (acurácia, precisão, recall, etc.), e analisar possíveis vieses ou limitações.

## Nível 0: Deploy

Preparar o modelo para ser implementado em produção. Isso inclui:

- Serialização do modelo (por exemplo, com `joblib` ou `pickle`)
- Construção de APIs para comunicação
- Monitoramento de performance e atualização de dados

### Exemplo de Deploy com `joblib`

```python
import joblib

# Salvar o modelo
joblib.dump(model, "meu_modelo.pkl")

# Carregar o modelo
modelo_carregado = joblib.load("meu_modelo.pkl")
```

---

# Modelos de Machine Learning

## Regressão Logística

A **regressão logística** é usada para tarefas de classificação binária. A função sigmoide transforma o output em uma probabilidade, geralmente para prever classes como 0 ou 1.

```python
from sklearn.linear_model import LogisticRegression

# Instanciar o modelo
logistic_model = LogisticRegression()

# Treinar o modelo
logistic_model.fit(X_train, y_train)

# Fazer predições
y_pred = logistic_model.predict(X_test)
```

## Support Vector Machines (SVM)

**SVM** é usado para tarefas de classificação, tentando maximizar a margem entre as classes. Pode ser usado para problemas lineares e não-lineares, usando funções de kernel.

```python
from sklearn.svm import SVC

# Instanciar o modelo
svm_model = SVC(kernel='linear')

# Treinar o modelo
svm_model.fit(X_train, y_train)

# Fazer predições
y_pred = svm_model.predict(X_test)
```

## Árvore de Decisão

**Árvores de decisão** dividem os dados em ramos baseados em condições, ideais para problemas de classificação e regressão.

```python
from sklearn.tree import DecisionTreeClassifier

# Instanciar o modelo
tree_model = DecisionTreeClassifier()

# Treinar o modelo
tree_model.fit(X_train, y_train)

# Fazer predições
y_pred = tree_model.predict(X_test)
```

## Modelos de Ensemble

Modelos de ensemble combinam múltiplos modelos para melhorar a performance. Os principais são:

### Bagging (Bootstrap Aggregating)

**Bagging** treina vários modelos independentes e combina seus resultados, ajudando a reduzir o overfitting. O **Random Forest** é um exemplo popular.

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Instanciar o modelo
rf_model = RandomForestClassifier(n_estimators=100)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer predições
y_pred = rf_model.predict(X_test)
```

### Boosting

**Boosting** treina modelos sequencialmente, onde cada modelo tenta corrigir os erros do anterior. Algoritmos como **Gradient Boosting** e **AdaBoost** são exemplos populares.

```python
from sklearn.ensemble import GradientBoostingClassifier

# Instanciar o modelo
gb_model = GradientBoostingClassifier()

# Treinar o modelo
gb_model.fit(X_train, y_train)

# Fazer predições
y_pred = gb_model.predict(X_test)
```

## Introdução a Redes Neurais

Redes neurais são compostas por camadas de neurônios e podem aprender padrões complexos. A biblioteca **Keras** é amplamente utilizada para construir redes neurais.

### Exemplo de Rede Neural em Python

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Construir o modelo
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # para classificação binária
])

# Compilar o modelo
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
nn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

# Exemplos Práticos para Métricas de Avaliação

## 1. Precisão (Precision)

### Exemplo
Imagine que temos um modelo de detecção de fraude. Queremos garantir que ele marque como fraudulenta apenas quando temos alta certeza para evitar falsos positivos.

```python
from sklearn.metrics import precision_score

# Suponha que temos as predições e os valores reais
y_true = [0, 1, 0, 1, 0, 1]  # Valores reais
y_pred = [0, 1, 0, 0, 0, 1]  # Predições do modelo

# Calcula a precisão
precision = precision_score(y_true, y_pred)
print("Precisão:", precision)
```

## 2. Recall

### Exemplo
Para uma aplicação médica, queremos identificar todos os casos positivos. O recall nos ajuda a garantir que estamos capturando a maioria dos positivos.

```python
from sklearn.metrics import recall_score

# Usando as mesmas predições e valores reais
recall = recall_score(y_true, y_pred)
print("Recall:", recall)
```

---

# Exemplo de Curva ROC e AUC

## Curva ROC e AUC

### Exemplo
A curva ROC nos ajuda a avaliar a performance do modelo em diferentes limiares de decisão.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Pontuações de probabilidade e valores reais
y_scores = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9]  # Exemplo de pontuações
y_true = [0, 0, 1, 1, 0, 1]

# Calcular taxa de verdadeiros e falsos positivos
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Linha para comparação aleatória
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
```

---

# Exemplo para Redes Neurais com Backpropagation

## Redes Neurais Multicamadas

### Exemplo
Construindo e treinando uma rede neural simples usando `Keras` para classificação binária.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configurar a rede neural
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))  # Primeira camada oculta
model.add(Dense(32, activation='relu'))               # Segunda camada oculta
model.add(Dense(1, activation='sigmoid'))             # Camada de saída para classificação binária

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
```

Este exemplo de rede neural demonstra o uso do backpropagation para ajustar os pesos ao longo do treino.

---

# Exemplo de Entropia e Índice de Gini

## Índice de Gini

### Exemplo
Usado para medir a impureza em uma árvore de decisão, o índice de Gini é uma métrica para decidir onde dividir os dados.

```python
from sklearn.tree import DecisionTreeClassifier

# Configuração da árvore com critério de Gini
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)
```

O índice de Gini ajuda a criar divisões nos dados para maximizar a separação das classes em cada nó da árvore.

---
