
# Processo de Modelagem de Machine Learning

## Nível 2: Escolha de Modelo

O primeiro passo no processo de modelagem em machine learning envolve a escolha cuidadosa do modelo adequado para o problema. A escolha do modelo leva em consideração aspectos como:

- **Tipo de Problema**: Determina-se se o problema é de classificação, regressão, agrupamento, entre outros.
- **Volume de Dados**: Alguns modelos são mais eficientes com grandes volumes de dados, enquanto outros funcionam melhor com datasets menores.
- **Complexidade dos Padrões**: Modelos mais simples, como regressões lineares, capturam padrões lineares, enquanto redes neurais podem modelar relacionamentos complexos e não-lineares.

### Divisão de Dados: Train, Validation e Test

Dividir os dados corretamente é uma prática fundamental em machine learning para evitar overfitting e garantir a generalização do modelo. A divisão em três subconjuntos principais é padrão:

- **Train (Treino)**: Parte dos dados usada para o modelo "aprender". Nesse conjunto, o modelo ajusta seus parâmetros internos para minimizar o erro.
- **Validation (Validação)**: Conjunto separado dos dados de treino para ajustar hiperparâmetros (parâmetros externos ao modelo). Ajuda a identificar quando o modelo está se ajustando demais aos dados de treino.
- **Test (Teste)**: Conjunto final que o modelo nunca viu antes, usado para medir sua performance final e garantir que ele possa generalizar bem para novos dados.

### Cross-Validation

A cross-validation (validação cruzada) é uma técnica usada para avaliar a estabilidade do modelo, especialmente quando os dados são limitados. Ela envolve dividir o conjunto de treino em várias "dobras" (folds) e alternar entre usar algumas para treino e uma para validação. Esse processo resulta em uma média de desempenho mais confiável e reduz o viés que pode ocorrer se usarmos apenas uma divisão simples.

**Vantagens do Cross-Validation**:
- **Melhora a robustez** dos resultados, já que o modelo é testado em diferentes subconjuntos de dados.
- **Reduz o overfitting**, já que o modelo é testado várias vezes e ajustado conforme necessário.
- **Permite otimização dos hiperparâmetros**, já que oferece uma visão mais completa de como o modelo responde a diferentes ajustes.

![image4.png](https://images.prismic.io/turing/6598098d531ac2845a272519_image5_11zon_af97fe4b03.webp?auto=format,compress)

## Nível 1: Certificação

Após o ajuste e avaliação inicial, é importante certificar que o modelo atende aos critérios de qualidade necessários para aplicação prática. A certificação envolve:

- **Avaliação de Métricas**: Análise de métricas de performance específicas, como acurácia, precisão, recall e F1-score, para problemas de classificação, ou erro médio quadrático (RMSE) para problemas de regressão.
- **Análise de Viés e Variância**: Avaliação para garantir que o modelo esteja equilibrado, evitando tanto o viés (underfitting) quanto a variância excessiva (overfitting).
- **Testes de Estresse e Segurança**: Simulação de cenários adversos para verificar a resiliência do modelo, incluindo dados fora da amostra e análise de possíveis vieses que possam afetar a justiça e a ética das previsões.

## Nível 0: Deploy

Após a certificação, o modelo é preparado para a produção. Esse processo envolve:

- **Serialização do Modelo**: O modelo é salvo em um formato adequado para ser facilmente carregado em ambientes de produção (ex.: `joblib` ou `pickle`), garantindo reprodutibilidade.
- **Criação de APIs**: Em ambientes modernos, os modelos são frequentemente disponibilizados por meio de APIs RESTful para facilitar a integração com outras aplicações.
- **Monitoramento Contínuo**: É importante monitorar o desempenho do modelo ao longo do tempo, pois as distribuições de dados podem mudar (data drift), o que pode exigir ajustes ou re-treinamento para manter a performance.

---

# Modelos de Machine Learning

## Regressão Logística

A **regressão logística** é uma técnica de classificação amplamente utilizada para problemas binários (duas classes). Embora tenha o nome "regressão", ela na verdade prevê probabilidades de uma amostra pertencer a uma determinada classe usando uma função sigmoide. Isso permite transformar a saída de uma regressão linear para um valor entre 0 e 1, facilitando a classificação em "sim" ou "não" (por exemplo, aprovado/reprovado, fraude/não fraude).

### Principais Características
- **Interpretação Probabilística**: A regressão logística permite prever probabilidades.
- **Linearidade**: A relação entre os atributos e a variável dependente é linear.
- **Usos**: Aplicada principalmente em classificação binária, mas pode ser estendida a problemas multiclasse.

## Support Vector Machines (SVM)

O **Support Vector Machines (SVM)** é um algoritmo robusto e versátil para classificação. Ele busca um "hiperplano ótimo" que maximiza a margem entre classes diferentes. A SVM é conhecida por sua capacidade de generalização e por ser menos sensível a dados ruidosos. A técnica de kernel permite que o SVM se adapte a problemas não-lineares.

### Principais Características
- **Máxima Margem**: Garante a máxima distância entre as classes, aumentando a separabilidade.
- **Flexibilidade com Kernel**: Pode lidar com problemas não-lineares usando diferentes funções de kernel (ex.: radial, polinomial).
- **Aplicações**: Muito utilizada em classificação de texto, imagem e problemas de bioinformática.

![images3.png](https://miro.medium.com/v2/resize:fit:876/0*UnQi3RThb0DroFH1.png)

## Árvore de Decisão

As **árvores de decisão** são modelos hierárquicos que dividem os dados com base em características específicas. Cada nó da árvore representa uma decisão baseada em um valor de atributo, enquanto as folhas representam a decisão final ou previsão. Árvores de decisão são intuitivas e interpretáveis, mas podem facilmente overfitar se não forem reguladas.

### Principais Características
- **Interpretação Simples**: A estrutura de árvore torna fácil visualizar e interpretar o processo de decisão.
- **Sensível ao Overfitting**: Árvores sem poda podem se ajustar excessivamente aos dados de treino.
- **Aplicações**: Usadas tanto em classificação quanto em regressão, principalmente quando interpretabilidade é crucial.

![image5.png](https://viso.ai/wp-content/uploads/2024/04/Visual-Representation-1-1060x596.png)

## Modelos de Ensemble

Modelos de ensemble combinam previsões de múltiplos modelos para obter um desempenho geral melhor. Eles podem reduzir a variabilidade e aumentar a precisão, sendo muito populares em competições e aplicações reais.

### Bagging (Bootstrap Aggregating)

No **Bagging**, vários modelos (geralmente fracos, como árvores de decisão) são treinados de forma independente e suas previsões são combinadas. A técnica ajuda a reduzir a variância e é menos propensa a overfitting. **Random Forest** é um exemplo de bagging popular, onde várias árvores de decisão são treinadas com diferentes subconjuntos de dados e combinadas para uma previsão robusta.

#### Principais Características do Random Forest
- **Redução de Variância**: A combinação de várias árvores reduz o risco de overfitting.
- **Generalização**: Funciona bem em vários tipos de problemas, mesmo com muitos atributos.
- **Aplicações**: Ampliamente usado em finanças, biomedicina e ciências sociais.

![image6.png](https://miro.medium.com/v2/resize:fit:696/1*7AxWhp2UMm7smg9ZmXlRHg.png)

### Boosting

**Boosting** é uma técnica em que modelos são treinados sequencialmente, com cada modelo tentando corrigir os erros do anterior. O algoritmo de boosting melhora a precisão ao aprender com os erros de modelos anteriores. Modelos como **Gradient Boosting** e **AdaBoost** são amplamente utilizados.

#### Características do Boosting
- **Foco em Erros**: Cada iteração se concentra nos erros dos modelos anteriores.
- **Robusto contra Outliers**: Pode ser ajustado para ser menos sensível a ruídos.
- **Aplicações**: Muito usado em desafios de machine learning, como classificação e predição de valores.

## Introdução a Redes Neurais

As redes neurais são compostas de camadas de neurônios artificiais, que se inspiram no funcionamento do cérebro humano. Cada "neurônio" realiza uma operação matemática e passa informações para outros neurônios em camadas subsequentes. Redes neurais são particularmente poderosas para tarefas complexas e não-lineares.

### Principais Características
- **Aprendizado Profundo**: Redes com múltiplas camadas podem aprender padrões complexos (Deep Learning).
- **Flexibilidade**: Capazes de resolver problemas em visão computacional, linguagem natural, e mais.
- **Exige Dados e Poder Computacional**: Treinamento pode ser custoso em termos de dados e recursos.

Ferramentas populares como **Keras** e **TensorFlow** facilitam a implementação de redes neurais e são amplamente usadas para tarefas complexas de machine learning.

---

Esse resumo oferece uma visão detalhada e teórica do processo de modelagem e dos principais modelos de machine learning. Ele é útil para compreender o raciocínio por trás das escolhas de modelo e das práticas recomendadas.


---

## 1. Precision (Precisão)

### Definição
A precisão é a medida que indica a proporção de **verdadeiros positivos** entre todos os resultados que o modelo previu como positivos. Em outras palavras, é a taxa de previsões corretas positivas em relação ao total de previsões positivas.

**seletividade**

### Fórmula
$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

### Exemplo
Um modelo de alta precisão é útil em situações onde se deseja minimizar falsos positivos. Por exemplo, um sistema de detecção de fraude em cartões de crédito deve ter alta precisão para evitar a marcação de transações legítimas como fraudulentas.

---

## 2. Recall (Revocação ou Sensibilidade)

### Definição
O recall mede a proporção de verdadeiros positivos em relação ao total de itens que realmente são positivos, ou seja, quantos dos positivos reais foram identificados corretamente.

**abrangência ou sensibilidade**

### Fórmula
$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

### Exemplo
Um modelo de alto recall é crucial em aplicações médicas para detectar doenças, pois é importante identificar todos os casos verdadeiros. Por exemplo, em um teste de câncer, é melhor ter um alto recall para evitar falsos negativos.

---

## 3. Specificity (Especificidade)

### Definição
A especificidade mede a capacidade de um modelo de identificar corretamente os **negativos verdadeiros**, ou seja, aqueles que realmente não são positivos.

(mesmo que Recall)

**abrangência, detecção para negativos**

### Fórmula
$$
\text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}}
$$

### Exemplo
A especificidade é importante em sistemas onde falsos positivos têm alto custo. No julgamento de um criminoso, é essencial ter uma alta especificidade para evitar condenações erradas.

---

## 4. Sensitivity (Sensibilidade)

### Definição
A sensibilidade, também chamada de **recall** em algumas áreas, mede a habilidade de um modelo em identificar corretamente todos os positivos verdadeiros.

(Recall dos negativos)

**abrangência, detecção**

### Fórmula
$$
\text{Sensitivity} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

### Exemplo
Em testes médicos, a sensibilidade é crucial para garantir que todos os casos da doença sejam detectados. Por exemplo, um teste para COVID-19 deve ter alta sensibilidade para minimizar o risco de falsos negativos.

---

## 5. Support Vector Machine (SVM)

### Definição
O Support Vector Machine (SVM) é um algoritmo de aprendizado supervisionado usado para classificação e regressão. O objetivo do SVM é encontrar um **hiperplano** que melhor separa as diferentes classes nos dados.

### Funcionamento
- O SVM encontra o hiperplano ideal que maximiza a **margem** entre as classes.
- Pontos de dados mais próximos do hiperplano, chamados de **vetores de suporte**, determinam a posição e orientação do hiperplano.

### Fórmula (Para um Hiperplano Linear)
Para um conjunto de dados em duas dimensões (x1 e x2), o hiperplano de separação pode ser representado pela equação:
$$
w_1 x_1 + w_2 x_2 + b = 0
$$
onde \( w_1 \) e \( w_2 \) são os pesos que definem a orientação do hiperplano, e \( b \) é o bias.

### Exemplo
O SVM é usado em classificações de texto, como filtragem de spam. Ele aprende a separar e-mails de spam dos e-mails legítimos com base em suas características (palavras, frequência de termos, etc.).

![images3.png](https://miro.medium.com/v2/resize:fit:876/0*UnQi3RThb0DroFH1.png)


---

# Curva ROC e AUC

---

## Curva ROC (Receiver Operating Characteristic)

### Definição
A curva ROC (Receiver Operating Characteristic) é uma ferramenta gráfica usada para avaliar a performance de um modelo de classificação binária. Ela plota a **taxa de verdadeiros positivos** (ou sensibilidade) contra a **taxa de falsos positivos** para diferentes limiares de decisão do modelo. 

### Funcionamento
- No eixo **Y**, temos a **taxa de verdadeiros positivos (TPR)** ou **sensibilidade**:
  $$
  \text{TPR} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  $$
- No eixo **X**, temos a **taxa de falsos positivos (FPR)**:
  $$
  \text{FPR} = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}
  $$
  
A curva ROC é traçada variando o limiar de decisão do modelo e medindo as taxas TPR e FPR correspondentes.

### Interpretação
- Quanto mais próxima a curva ROC está do canto superior esquerdo (ponto onde a TPR é 1 e a FPR é 0), melhor é o desempenho do modelo.
- A linha diagonal (de 0,0 a 1,1) representa o desempenho de um modelo aleatório. Quanto mais distante a curva ROC estiver dessa linha, melhor é o modelo.


(False positive rate = 1 - especifidade)

![image.png](https://i0.wp.com/sefiks.com/wp-content/uploads/2020/12/roc-curve-original.png?fit=726%2C576&ssl=1)

---

## AUC (Area Under the Curve)

### Definição
A AUC (Area Under the Curve) é uma métrica que calcula a **área sob a curva ROC**. Esse valor representa a capacidade do modelo de classificar corretamente os positivos e negativos, independentemente do limiar.

### Interpretação
- A **AUC varia de 0 a 1**. Quanto maior o valor da AUC, melhor o modelo é para distinguir entre as classes.
- **Valores da AUC**:
  - **AUC = 0.5**: O modelo tem desempenho equivalente ao acaso, ou seja, não consegue distinguir as classes.
  - **AUC entre 0.5 e 0.7**: Desempenho pobre do modelo.
  - **AUC entre 0.7 e 0.8**: Modelo com desempenho razoável.
  - **AUC entre 0.8 e 0.9**: Modelo com bom desempenho.
  - **AUC entre 0.9 e 1**: Modelo com excelente desempenho.
  
### Valor Ideal de AUC
O valor ideal de AUC é **1**, o que indica que o modelo consegue separar perfeitamente as classes sem cometer erros. No entanto, na prática, valores entre **0.8 e 0.9** já são considerados bons para muitos casos de uso.

---

## Exemplo de Uso da AUC
A AUC é especialmente útil em problemas onde temos que balancear o custo de falsos positivos e falsos negativos. Por exemplo:
- **Na detecção de fraudes**: Um modelo com alta AUC ajudaria a identificar transações fraudulentas de forma confiável.
- **Em diagnósticos médicos**: Um teste com alta AUC garante que a maioria dos pacientes com uma condição sejam corretamente identificados.

---

## Fórmulas

- **Taxa de Verdadeiros Positivos (TPR)** ou **Sensibilidade**:
  $$
  \text{TPR} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  $$

- **Taxa de Falsos Positivos (FPR)**:
  $$
  \text{FPR} = \frac{\text{False Positives (FP)}}{\text{False Positives (FP)} + \text{True Negatives (TN)}}
  $$

A **AUC** é a **área sob a curva** ROC, mas não possui uma fórmula exata, pois depende da forma da curva. Em geral, a AUC é calculada numericamente a partir dos pontos da curva ROC.

![image2.png](https://cdn.prod.website-files.com/660ef16a9e0687d9cc27474a/662c42679571ef35419c9968_64760779d5dc484958a3f917_classification_metrics_017-min.png)
---

# Backpropagation e Redes Neurais Multicamadas

## Redes Neurais Multicamadas (Multilayer Neural Networks)
Uma rede neural multicamada (também chamada de Multilayer Perceptron, ou MLP) é composta por várias camadas de neurônios:
- **Camada de Entrada**: Recebe os dados iniciais e os passa para as próximas camadas.
- **Camadas Ocultas**: Essas camadas processam e transformam os dados de entrada. Em redes multicamadas, é comum ter várias dessas camadas. Cada neurônio em uma camada está conectado a neurônios na camada seguinte.
- **Camada de Saída**: Fornece a previsão ou a classificação final, dependendo do problema (regressão ou classificação).

Cada conexão entre neurônios possui um peso que define a força ou importância dessa conexão. Durante o treinamento, esses pesos são ajustados para minimizar o erro entre a previsão da rede e o valor esperado.

## Backpropagation
Backpropagation (ou retropropagação) é o algoritmo principal para treinar redes neurais multicamadas. Ele ajusta os pesos da rede para minimizar o erro entre a saída prevista e a saída esperada, aplicando o **gradiente descendente**.

### Etapas do Backpropagation
1. **Forward Pass**: 
   - Os dados de entrada são passados pela rede, camada por camada, até a camada de saída, onde uma previsão é gerada.

2. **Cálculo do Erro**:
   - O erro é calculado comparando a saída prevista com o valor esperado. Funções de erro comuns incluem o **erro quadrático médio** e a **entropia cruzada**.

3. **Backward Pass (Retropropagação)**:
   - O erro calculado é então propagado para trás através da rede para atualizar os pesos. O objetivo é diminuir o erro, de modo que a rede produza previsões mais precisas.

4. **Atualização dos Pesos**:
- O algoritmo usa o **gradiente descendente** para ajustar os pesos, reduzindo o erro ao longo de várias iterações.
- Para cada peso \( w \), ele é ajustado usando a fórmula:
  $$
  w = w - \alpha \frac{\partial E}{\partial w}
  $$
  onde:
  - $\alpha $ é a taxa de aprendizado,
  - $ E $ é o erro da rede,
  - $ \frac{\partial E}{\partial w} $ é o gradiente do erro em relação ao peso $ w $.



### Vantagens e Desafios
- **Vantagens**: O backpropagation permite que redes complexas com muitas camadas aprendam representações complexas dos dados, tornando-as altamente eficazes para problemas como reconhecimento de imagem e processamento de linguagem natural.
- **Desafios**: Treinar redes profundas pode ser demorado e requer um bom ajuste da taxa de aprendizado e outras hiperparâmetros.

### Exemplo Visual
Para entender melhor, imagine uma rede neural com 3 camadas (entrada, uma oculta e saída). Ao treinar a rede com backpropagation, os pesos entre cada camada são ajustados para que a saída se aproxime do valor esperado após várias iterações.

---

Backpropagation, junto com redes neurais multicamadas, é fundamental para o funcionamento da maioria das redes neurais modernas. Com redes profundas e o ajuste dos pesos usando gradiente descendente, podemos resolver problemas complexos de aprendizado de máquina.


# Função de Perda de Entropia Cruzada

A **entropia cruzada** é uma função de perda amplamente usada em problemas de classificação, especialmente em redes neurais. Ela mede a discrepância entre a distribuição de probabilidade real (rótulos) e a distribuição de probabilidade prevista pela rede.

Para um problema de classificação binária, onde:
- $ y $ é o rótulo verdadeiro, que pode ser 0 ou 1,
- $ \hat{y} $ é a probabilidade prevista para a classe positiva (geralmente saída de uma função sigmoide),

a função de perda de entropia cruzada é dada por:

$$
L = - \left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)
$$

### Explicação dos Termos
- $ y \cdot \log(\hat{y}) $: Penaliza a perda quando o rótulo verdadeiro é 1, e a rede prevê uma probabilidade baixa para a classe positiva.
- $ (1 - y) \cdot \log(1 - \hat{y}) $: Penaliza a perda quando o rótulo verdadeiro é 0, e a rede prevê uma probabilidade alta para a classe positiva.

### Para Classificação Multiclasse
Para problemas de múltiplas classes (com $ C $ classes), onde $ yi $ é o vetor de rótulos verdadeiros em codificação one-hot e $ \hat{y}_i $ é a probabilidade prevista para a classe $ i $, a entropia cruzada é:

$$
L = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
$$

Essa fórmula calcula a soma das perdas individuais para cada classe, ponderada pela presença de cada classe no rótulo real.

A entropia cruzada é uma função de perda eficaz porque amplifica os erros nas previsões de probabilidade, incentivando a rede a aprender representações mais precisas para as classes.



# Entropia e Índice de Gini

A **entropia** e o **índice de Gini** são medidas de impureza comumente usadas em algoritmos de aprendizado de máquina, especialmente em árvores de decisão, para avaliar a divisão de dados e a separação de classes. Ambos medem a "impureza" ou a "diversidade" em uma amostra, ajudando a definir onde e como dividir os dados para obter a melhor classificação.

## Entropia

A **entropia** mede o grau de incerteza ou aleatoriedade de uma distribuição de probabilidade. Em aprendizado de máquina, é usada para quantificar a impureza de uma amostra em um conjunto de dados. Quanto maior a entropia, maior a desordem ou impureza do conjunto.

Para um conjunto de dados com duas classes (por exemplo, positivo e negativo), a entropia \( E \) é dada por:

$$
E = - (p_+ \cdot \log(p_+) + p_- \cdot \log(p_-))
$$

Onde:
- \( p_+ \) é a proporção de elementos positivos na amostra,
- \( p_- \) é a proporção de elementos negativos na amostra.

### Explicação dos Termos
- \( p_+ \cdot \log(p_+) \): Penaliza a impureza quando há uma alta incerteza sobre a classe positiva.
- \( p_- \cdot \log(p_-) \): Penaliza a impureza quando há uma alta incerteza sobre a classe negativa.

Quando a entropia é 0, significa que todos os elementos estão na mesma classe (impureza mínima). Quando a entropia é máxima, há uma distribuição uniforme entre as classes (alta impureza).

## Índice de Gini

O **índice de Gini** também mede a impureza de uma amostra, mas de uma maneira ligeiramente diferente da entropia. O índice de Gini é frequentemente usado em algoritmos como o CART (Classification and Regression Tree) para decidir a melhor divisão dos dados. 

Para um conjunto de dados com duas classes, o índice de Gini $ G $ é dado por:

$$
G = 1 - (p_+^2 + p_-^2)
$$

Onde:
- $ p_+ $ é a proporção de elementos positivos na amostra,
- $ p_- $ é a proporção de elementos negativos na amostra.

### Explicação dos Termos
- $ p_+^2 $ e $ p_-^2 $: Representam as probabilidades de uma amostra ser classificada corretamente dentro da classe positiva e negativa, respectivamente.

O índice de Gini varia de 0 a 0,5 para uma classificação binária. Quanto mais próximo de 0, menor a impureza. Quanto mais próximo de 0,5, maior a impureza, indicando uma distribuição mais uniforme entre as classes.

### Comparação entre Entropia e Índice de Gini
Ambas as métricas são usadas para medir impureza, mas a entropia tende a penalizar mais incertezas extremas, enquanto o índice de Gini é mais simples computacionalmente. Em muitos casos, ambos fornecem resultados semelhantes ao decidir divisões em árvores de decisão, embora a escolha entre eles possa influenciar ligeiramente a estrutura da árvore.

