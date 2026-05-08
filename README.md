# Proposta de TCC: Engenharia de ML e Observabilidade

## 1. Definição do Projeto

### 1.1. Título Sugerido
**"Arquitetura Cross-Engine para Observabilidade de Modelos de Machine Learning: Implementação de uma Biblioteca Agnosticista com Foco em Eficiência de Processamento Distribuído."**

### 1.2. Objeto de Estudo
O projeto consiste no desenvolvimento da **LinceML**, uma biblioteca em Python projetada para monitorar a saúde de modelos de Machine Learning em produção. A ferramenta diferencia-se das soluções de mercado ao permitir a execução de cálculos estatísticos de *Drift* (Data, Model e Concept) de forma nativa em diferentes motores de processamento (**Polars** e **Apache Spark**), sem a necessidade de extração de dados (*data-out*) para o ambiente de execução do driver.

### 1.3. O Problema (Problemática)
Embora a Inteligência Artificial seja vital para o crescimento empresarial, a realidade da implementação é marcada por desafios que levam a falhas significativas. Dados recentes mostram uma realidade severa:

1. **Altas Taxas de Insucesso:** Em 2022, cerca de 80% dos projetos de IA não atingiram seus objetivos, com muitos sequer ultrapassando a fase piloto.
2. **Persistência do Cenário:** Em 2023, aproximadamente 75% das iniciativas falharam ou ficaram abaixo das expectativas.
3. **Barreiras Técnicas e de Talento:** Fatores críticos para esse insucesso incluem a má gestão de dados, falta de talentos especializados e a superestimação das capacidades da IA.

No cenário atual de MLOps, a observabilidade de modelos enfrenta três grandes gargalos que este trabalho visa endereçar:

1. **O Abismo Local vs. Produção:** Ferramentas populares (como *Evidently AI* ou *Pandas-based profilers*) funcionam bem em protótipos locais, mas falham ou tornam-se proibitivamente lentas ao lidar com o volume de dados de produção em ambientes de Big Data (clusters Spark).
2. **O Gargalo do Data-Out:** A prática comum de converter DataFrames distribuídos em objetos locais (`.toPandas()`) para cálculos estatísticos gera latência excessiva, custos de rede e riscos de falhas por falta de memória (*Out of Memory - OOM*).
3. **Inchaço de Artefatos (Metadata Bloat):** Muitas ferramentas geram relatórios JSON ou HTML pesados que dificultam o armazenamento histórico e a análise de séries temporais de longa duração, essencial para detectar *Concept Drift* em janelas mensais.

#### 1.3.1. O Abismo Local vs. Produção (A Falha na Escala)

1. **O Dado:** De acordo com o VentureBeat, cerca de 87% dos projetos de Machine Learning nunca chegam à produção. O principal motivo não é a falta de acurácia, mas a incapacidade de integrar o modelo à infraestrutura de dados existente (Escalabilidade).

2. **A Prova Técnica:** O Pandas, motor de ferramentas como o Evidently, é single-threaded e opera inteiramente na RAM. Em datasets acima de 10GB, o Pandas frequentemente atinge o limite de swap ou falha catastróficamente, enquanto o Polars ou Spark processam o mesmo volume com eficiência linear.

3. **Argumento de Valor:** "Ferramentas que não suportam processamento distribuído nativo criam um 'teto de vidro' para a observabilidade, limitando-a a amostras estatísticas que podem mascarar anomalias em cauda longa (long-tail anomalies)."

#### 1.3.2. O Gargalo do Data-Out (O Custo da Movimentação)

1. **O Dado:** Estudos de engenharia de Big Data indicam que a Serialização e Deserialização (SerDe) pode consumir até 80% do tempo total de processamento em sistemas distribuídos. Quando você faz um `.toPandas()`, você está pagando esse "pedágio" desnecessariamente.

2. **A Prova Técnica:** O custo de rede e o risco de Out of Memory (OOM) no Driver do Spark. Se o Driver tentar coletar um volume de dados para calcular um PSI (Population Stability Index) localmente, o cluster inteiro fica ocioso enquanto um único nó tenta processar o dado, criando um ponto único de falha.

3. **Argumento de Valor:** "O processamento in-situ (dentro do Spark) não é apenas uma questão de velocidade, mas de conformidade e segurança: quanto menos o dado transita pela rede, menor a superfície de exposição e o custo de transferência (egress costs)."

#### 1.3.3. Inchaço de Artefatos (Metadata Bloat)

1. **O Dado:** Um relatório do Evidently AI para 1 milhão de predições pode gerar um arquivo JSON/HTML de 5MB a 20MB. Em um pipeline que roda a cada hora por um ano, isso gera cerca de 175GB de metadados apenas para um modelo.

2. **A Prova Técnica:** Comparação de densidade de dados.

* **Abordagem Tradicional:** Salvar logs brutos ou JSONs densos.

* **Abordagem da LinceML:** Uso de Statistical Sketches. Um histograma ou um T-Digest de 1 bilhão de registros ocupa menos de 100KB.

---

### 1.4. Pilares Centrais
A proposta fundamenta-se em três pilares técnicos de alto valor:

* **Processamento In-Situ (Nativo):** No motor Spark, a biblioteca utiliza apenas a API nativa de funções do Spark SQL para realizar cálculos, garantindo que o processamento ocorra onde o dado reside.
* **Dualidade de Performance:** Suporte ao **Polars** para processamento local ultra-rápido (utilizando Rust por baixo) e ao **Spark** para escalabilidade horizontal massiva.
* **Flexibilidade Configurável:** O usuário tem o controle explícito sobre a estratégia de dados (`allow_data_out=True/False`), permitindo um equilíbrio entre a precisão de testes estatísticos complexos e a performance de infraestrutura.

---

### 1.5. Valor de Negócio (Visão Executiva)
Para uma instituição financeira ou empresa de tecnologia, a adoção de uma ferramenta com esta arquitetura resulta em:
* **Redução de Custo de Cloud:** Menor tráfego de dados e uso otimizado de instâncias de computação.
* **Confiabilidade do Modelo:** Detecção precoce de degradação de performance, permitindo retreinos assertivos.
* **Sustentabilidade (Green IT):** Processamento eficiente que reduz a pegada de carbono da infraestrutura de ML.

---

## 2. Fundamentação e Problematização

### 2.1. O "Gap" de Sustentação no Ciclo de MLOps
O ciclo de vida de um modelo de Machine Learning é frequentemente dividido em *Build* (treinamento) e *Run* (operação). Enquanto a academia e o mercado focam massivamente em algoritmos de treinamento, a sustentação de modelos em produção revela um "débito técnico" oculto. 

A maioria dos modelos degrada no momento em que entra em contato com dados reais, fenômeno conhecido como *Drift*. A carência de ferramentas que unam a facilidade de desenvolvimento local com a robustez de ambientes distribuídos cria um vácuo operacional onde modelos falham silenciosamente por meses antes de serem detectados.

### 2.2. Limitações das Ferramentas de Mercado (State-of-the-Art Analysis)
Embora existam bibliotecas consolidadas, sua aplicação em cenários de Big Data e alta conformidade (como o setor bancário) apresenta limitações críticas:

* **Acoplamento ao Driver (Gargalo de Memória):** Ferramentas como o *Evidently AI* dependem da conversão de dados para objetos Python/Pandas. Em datasets de escala terabyte, isso torna o monitoramento impraticável ou extremamente caro, exigindo máquinas de driver com centenas de GBs de RAM.
* **Custo de Armazenamento de Metadados:** O output padrão de muitas ferramentas são arquivos HTML ou JSONs densos. Para um monitoramento histórico (mensal/anual), o custo de armazenar esses artefatos "inchados" supera o valor gerado pela observabilidade.
* **Rigidez Estatística:** Muitas libs são "caixas-pretas" que dificultam a implementação de métricas customizadas exigidas por órgãos reguladores ou necessidades específicas de negócio.

### 2.3. Justificativa Técnica: A Abordagem Cross-Engine
A escolha técnica deste projeto fundamenta-se na necessidade de **portabilidade estatística**. 

1.  **Por que Polars?** Representa o estado da arte em processamento local, oferecendo performance superior ao Pandas através de execução em Rust e processamento *lazy-evaluation*, ideal para ciclos rápidos de desenvolvimento e validação.
2.  **Por que Spark Native?** Em ambientes corporativos, o dado raramente pode ou deve sair do cluster. Implementar o monitoramento através da API nativa do Spark elimina o risco de *Data-out* e aproveita a escalabilidade horizontal já paga pela infraestrutura de dados.

### 2.4. Taxonomia do Drift no Escopo do Projeto
Para garantir a "qualidade indiscutível", a biblioteca abordará as três dimensões fundamentais da degradação:

* **Data Drift:** Mudanças na distribuição estatística das *features* de entrada ($P(X)$).
* **Concept Drift:** Mudança na relação entre as variáveis e o alvo ($P(Y|X)$), exigindo tratamento de rótulos atrasados (*delayed labels*).
* **Quality Drift:** Degradação técnica, como aumento de valores nulos, erros de tipagem ou quebras de contrato de dados.

---

## 3. Arquitetura da Biblioteca (Software Engineering)

A arquitetura da **LinceML** é desenhada sob o princípio da "Inversão de Dependência", garantindo que a lógica estatística seja independente da tecnologia de processamento de dados. O objetivo é que o usuário final interaja com uma API única, enquanto o "motor" sob o capô se adapta ao volume de dados e à infraestrutura disponível.

### 3.1. Padrões de Projeto e Abstração
Para garantir a extensibilidade e a paridade de resultados entre diferentes backends, a biblioteca utiliza os seguintes padrões:

* **Strategy Pattern:** Utilizado para alternar entre os motores de execução (`PolarsEngine` vs. `SparkEngine`) em tempo de execução, baseando-se na tipagem do dado de entrada ou configuração explícita.
* **Adapter Pattern:** Traduz as operações estatísticas abstratas (ex: `compute_histogram`) para as funções específicas de cada framework (ex: `.value_counts()` no Polars ou `.groupBy().count()` no Spark).

### 3.2. Estrutura de Motores (Engines)

#### 3.2.1. Motor Polars (Local/Single-Node)
Focado em alta performance para cientistas de dados em ambiente de desenvolvimento ou modelos com volumetria moderada.
* **Vantagem:** Utiliza *multithreading* nativo em Rust e *Memory Mapping* (Apache Arrow).
* **Abordagem:** Execução *Eager* ou *Lazy*, otimizando o plano de execução antes de materializar os cálculos de drift.

#### 3.2.2. Motor Spark (Distribuído/Big Data)
Desenhado para rodar em clusters (ex: EMR, Databricks) onde o dado é distribuído entre diversos workers.
* **Estratégia Native-First:** Todas as métricas básicas (Média, Desvio Padrão, Percentis) são calculadas via `pyspark.sql.functions`, aproveitando o *Catalyst Optimizer*.
* **Resiliência:** O motor é construído para evitar *shuffles* desnecessários, mantendo a localidade do dado sempre que possível.

### 3.3. Gerenciamento de Fluxo de Dados (Mecanismo Data-Out)
Um dos maiores diferenciais técnicos da biblioteca é o controle granular sobre como o dado transita entre o Cluster e o Driver Python.

| Configuração | Comportamento no Spark | Caso de Uso |
| :--- | :--- | :--- |
| `allow_data_out=False` | **Cálculo Distribuído:** Apenas o resultado estatístico final (ex: o valor do PSI) é retornado ao Driver. | Produção, Big Data, Compliance Bancário. |
| `allow_data_out=True` | **Coleta Parcial/Total:** O dado é coletado para o Driver para execução de testes via SciPy/NumPy. | Análises exploratórias, Testes de hipóteses complexos. |

### 3.4. Contratos e Saídas (Payloads)
Para garantir que a biblioteca seja integrável em pipelines de CI/CD e ferramentas de monitoramento (como Grafana ou Datadog), os resultados seguem um contrato rígido:
* **Schema Unificado:** Independente do motor, o output é um objeto `DriftReport` validado via **Pydantic**.
* **Formatos de Exportação:** Suporte nativo para JSON (leve, focado em logs) e dicionários Python para integração imediata com sistemas de alerta.

### 3.5. Ciclo de Vida e Empacotamento
Como uma biblioteca voltada para engenheiros, a **LinceML** adotará padrões modernos de distribuição:
* **Gestão de Dependências:** Uso de `Poetry` ou `uv` para garantir builds reprodutíveis.
* **Qualidade de Código:** Suite de testes automatizados com `Pytest`, utilizando o `Toxic` para testar a compatibilidade entre diferentes versões de Python e Spark.
* **Documentação:** Documentação técnica autogerada via `MkDocs` com exemplos de *docstrings* seguindo o padrão Google.

---

## 4. Motor Estatístico e Funcionalidades (O "Coração")

O diferencial da **LinceML** reside na implementação manual de testes estatísticos otimizados para execução distribuída. Ao evitar bibliotecas de alto nível (como *Scikit-Learn* ou *Evidently*) dentro do motor Spark, garantimos a execução via **Spark Functions**, o que elimina o overhead de serialização e movimentação de dados.

### 4.1. Módulo de Data Quality (Sanity Checks)
Antes do cálculo de drift, a biblioteca realiza uma validação de integridade para evitar falsos positivos causados por erros de pipeline:
* **Schema Enforcement:** Comparação de tipos primitivos entre o conjunto de referência (treino) e o de análise (produção).
* **Null Rate Monitoring:** Cálculo da variação percentual de valores nulos por coluna.
* **Range Validation:** Detecção de valores fora dos limites esperados (min/max) definidos no treinamento.

### 4.2. Detecção de Data Drift (Estabilidade de Features)
A biblioteca foca em métricas que quantificam a divergência entre a distribuição de treino ($P$) e a de produção ($Q$).

#### 4.2.1. Population Stability Index (PSI)

$$PSI = \sum_{i=1}^{B} \left( \%Actual_i - \%Expected_i \right) \cdot \ln \left( \frac{\%Actual_i}{\%Expected_i} \right)$$
* **Implementação Spark:** Os dados são particionados em *decis* (buckets) utilizando a função `percent_rank()`, e as contagens são agregadas via `groupBy`, resultando em um payload final de apenas alguns bytes enviado ao Driver.

#### 4.2.2. Divergência de Jensen-Shannon (JSD)
Utilizada para medir a similaridade entre distribuições de probabilidade, sendo uma versão simétrica e suavizada da divergência de Kullback-Leibler.
$$JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)$$
Onde $M = \frac{1}{2}(P + Q)$. Esta métrica é ideal para monitoramento contínuo por ser limitada entre 0 e 1.

#### 4.2.3. Teste Kolmogorov-Smirnov (KS)
O teste KS é utilizado para comparar as distribuições de probabilidade de duas amostras contínuas, sendo uma ferramenta essencial para detectar desvios de forma e localização nas *features*. 

A estatística $D$ é definida pelo supremo da distância absoluta entre as Funções de Distribuição Acumulada (ECDF) da amostra de referência ($F_{ref}$) e da amostra de produção ($F_{prod}$):

$$D_{n,m} = \sup_{x} |F_{ref,n}(x) - F_{prod,m}(x)|$$

* **Implementação em Spark (No Data-Out):** Para evitar a coleta de dados para o Driver, a biblioteca utiliza funções de janela (`Window`) e a função `percent_rank()` para computar a ECDF de forma distribuída nos *workers*. Isso garante que a volumetria massiva seja processada sem o risco de falhas por estouro de memória, mitigando um dos principais fatores técnicos de insucesso em projetos de IA[cite: 80, 199].
* **Implementação em Polars:** Aproveita o processamento em Rust para realizar a ordenação e o cálculo da distância máxima de forma vetorizada, garantindo agilidade no ambiente de desenvolvimento local.

#### 4.2.4. Teste Qui-Quadrado ($\chi^2$)
Para variáveis categóricas, onde o teste KS não é aplicável, a **LinceML** implementa o teste de aderência Qui-Quadrado. Ele avalia se a frequência observada em produção ($O_i$) diverge significativamente da frequência esperada baseada no treino ($E_i$):

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

* **Implementação em Spark:** O motor utiliza agregações nativas (`groupBy` e `count`) para gerar as tabelas de contingência. O cálculo final da estatística é realizado sobre os agregados, mantendo o tráfego de rede mínimo e alinhando a infraestrutura aos objetivos estratégicos de eficiência operacional.
* **Monitoração Contínua:** A automação destes testes permite o estabelecimento de loops de feedback constantes, uma prática recomendada para adaptar modelos a ambientes de negócios dinâmicos e evitar a obsolescência das predições.

### 4.3. Concept Drift e Qualidade do Modelo (Delayed Labels)
Diferente do Data Drift, o Concept Drift exige o retorno da realidade (*Ground Truth*), que em muitos casos (como crédito ou fraude) ocorre com atraso.

* **Arquitetura de Reconciliação:** A biblioteca provê um mecanismo para unir predições históricas com labels reais através de chaves únicas (*join* distribuído no Spark).
* **Métricas de Performance:** Cálculo de matriz de confusão, *Precision*, *Recall*, *F1-Score* e *ROC-AUC*.
* **Análise de Resíduos:** Para modelos de regressão, monitoramento da distribuição do erro absoluto para detectar se o modelo está subestimando ou superestimando previsões sistematicamente.

### 4.4. Otimização via Statistical Sketches
Para viabilizar a observabilidade de longo prazo sem custos explosivos de storage, a **LinceML** implementa o conceito de "Resumos Estatísticos":

1. **Agregação em Tempo de Processamento:** Em vez de salvar cada predição, o motor calcula histogramas, momentos (média, variância) e quantis.
2. **Compactação de Artefatos:** O resultado de um processamento de 1 TB de dados é convertido em um objeto de metadados de poucos KBs (Statistical Sketch).
3. **Comparabilidade Histórica:** Estes sketches podem ser comparados entre si (ex: Janeiro vs. Fevereiro) sem a necessidade de reprocessar os dados brutos originais.

### 4.5. Implementação "From Scratch" no Spark
O "pulo do gato" técnico desta seção é o uso de **Aggregators Customizados** ou expressões puras de Spark SQL para métricas complexas. 
* **Exemplo:** O cálculo do teste Kolmogorov-Smirnov (KS) é decomposto em funções de Distribuição Acumulada (CDF) calculadas via *Window Functions*, evitando o uso de `collect()` e mantendo o paralelismo total do cluster.

---

## 5. Metodologia de Validação (A "Arena" de Benchmarking)

Para validar a qualidade e a eficácia da **LinceML**, o trabalho adotará uma abordagem experimental rigorosa, comparando a biblioteca desenvolvida com os principais *players* de mercado (*Evidently AI* e *WhyLogs*). O objetivo é demonstrar que a solução customizada mantém a precisão estatística enquanto reduz drasticamente o consumo de recursos computacionais.

### 5.1. Ambiente Experimental
A validação será conduzida em dois ambientes distintos para testar a premissa de dualidade da biblioteca:
1.  **Ambiente Local (Small Data):** Máquina com 16GB RAM, processador de 8 cores, utilizando o motor **Polars**.
2.  **Ambiente Distribuído (Big Data):** Cluster Apache Spark emulado via Docker (1 Master, 3 Workers) com restrição proposital de memória no Driver para testar a resiliência ao *Data-Out*.

### 5.2. Dataset e Simulação de Drift
Será utilizado um dataset público clássico de séries temporais (ex: *NYC Taxi Dataset* ou *Forest Covertype*) ou um dataset sintético controlado, onde o drift será injetado artificialmente em três cenários:
* **Cenário A (Sudden Drift):** Mudança abrupta em uma feature numérica (ex: multiplicação por um fator de escala).
* **Cenário B (Quality Drift):** Introdução gradual de 20% de valores nulos em colunas críticas.
* **Cenário C (Concept Drift):** Alteração na lógica do alvo ($y$) após $T$ meses para validar o motor de *Delayed Labels*.

### 5.3. Critérios de Comparação (Métricas de Sucesso)
A "Qualidade Indiscutível" será provada através de quatro eixos métricos:

| Métrica | O que avalia? | Meta (KR) |
| :--- | :--- | :--- |
| **Paridade Estatística** | O valor do PSI/KS calculado pela LinceML vs. Bibliotecas padrão (SciPy). | Erro relativo < 0,001% |
| **Peak RAM (Driver)** | O consumo máximo de memória no nó principal durante o processamento de 10GB+ de dados. | < 20% do consumo do Evidently AI |
| **Artifact Size** | O tamanho final do arquivo (JSON/Parquet) que armazena os metadados do monitoramento. | < 500KB por batch de 1M linhas |
| **Execution Time** | Tempo total desde a leitura do dado bruto até a geração do relatório de drift. | Superior ao Spark nativo com .toPandas() |

### 5.4. Procedimento de Teste "No Data-Out"
Um teste de estresse específico será realizado no motor Spark:
1.  Configurar o Driver do Spark com apenas 1GB de RAM.
2.  Processar um volume de dados de 50GB.
3.  **Hipótese:** A *LinceML* (com `allow_data_out=False`) completará a tarefa com sucesso, enquanto ferramentas que dependem de coleta para o driver falharão com erro de *Out Of Memory (OOM)*.

### 5.5. Avaliação da Simplicidade e Flexibilidade
Além dos dados quantitativos, o trabalho apresentará uma análise qualitativa baseada na "Experiência do Desenvolvedor" (DX):
* **Linhas de Código (LoC):** Quantidade de código necessário para configurar o monitoramento em um novo pipeline.
* **Portabilidade:** Demonstração do mesmo script de monitoramento rodando em Polars (local) e Spark (cluster) apenas alterando uma flag de configuração.

---

## 6. OKRs e Metas de Sucesso (Valoração de Valor)

Para garantir que o TCC atinja uma "ótima qualidade", o sucesso do projeto não será medido apenas pela entrega do código, mas pelo alcance de metas reais e mensuráveis, estruturadas através do framework de OKRs (*Objectives and Key Results*).

### Objetivo 1: Excelência em Engenharia e Portabilidade Cross-Engine
**Meta:** Demonstrar que a biblioteca é capaz de transitar entre ambientes de desenvolvimento e produção sem perda de consistência ou necessidade de refatoração.
* **KR 1:** Alcançar **Paridade de Resultados** com erro relativo $< 0,001\%$ entre as execuções em Polars e Spark para as mesmas métricas estatísticas.
* **KR 2:** Garantir que o mesmo script de monitoramento execute em ambos os motores alterando apenas a flag de configuração de entrada.
* **KR 3:** Manter a cobertura de testes unitários (**Code Coverage**) acima de 90%, validando todos os motores de execução.

### Objetivo 2: Eficiência de Infraestrutura e Custo Operacional
**Meta:** Provar a superioridade da abordagem de processamento nativo (sem data-out) em relação ao estado da arte.
* **KR 1:** Reduzir em pelo menos **80% o volume de dados trafegados** (network I/O) no cluster Spark ao utilizar o modo `allow_data_out=False`.
* **KR 2:** Garantir que o artefato de monitoramento final (Metadata Sketch) seja pelo menos **10x menor** que o output padrão do *Evidently AI* para o mesmo dataset.
* **KR 3:** Processar um dataset que exceda em 5x a memória disponível do Driver no Spark sem ocorrência de falhas por *Out Of Memory* (OOM).

### Objetivo 3: Documentação e Experiência do Desenvolvedor (DX)
**Meta:** Facilitar a adoção da biblioteca por outros engenheiros de ML.
* **KR 1:** Disponibilizar documentação técnica completa via *MkDocs*, incluindo um guia de "Quick Start" que permita a primeira detecção de drift em menos de 5 minutos.
* **KR 2:** Implementar tipagem estática rigorosa com *Python Type Hints* e validação de parâmetros com *Pydantic* em 100% da API pública.

---

## 7. Cronograma e Planejamento de Execução

O trabalho será executado em um ciclo de 4 meses, seguindo uma abordagem de desenvolvimento iterativo (Agile), permitindo que a escrita da tese ocorra em paralelo com a codificação da biblioteca.

### 7.1. Cronograma Detalhado

| Fase | Atividades Principais | Prazo (Mês.Semana) |
| :--- | :--- | :--- |
| **Fase 1: Setup** | Cenários de teste, casos de uso e base de dados | M1.S1 | 
| | estrutura da biblioteca, pipeline de testes | M1.S2 |
| **Fase 2: Core** | Design da API, implementação da BaseEngine. | M1.S4 |
| | Criação e execução dos scripts de teste para o BaseEngine | M2.S1 |
| **Fase 3: Motor Polars** | Desenvolvimento das funções nativas no Polars e lógica de "Lazy/Eagle Process". | M2.S3 |
| | Criação e execução dos scripts de teste para o BaseEngine | M2.S4 |
| **Fase 4: Motor Spark** | Desenvolvimento das funções nativas no Spark e lógica de "Allow Data Out". | M3.S2 |
| | Criação e execução dos scripts de teste para o BaseEngine | M3.S3 |
| **Fase 5: Métricas e Drift** | Implementação de PSI e KS-Test | M4.S2 |
| | Implementação de Jesen-Shannon e Chi-square | M5.S1 |
| | Implementação de lógica de reconciliação de labels atrasados. | M5.S4 |
| **Fase 6: Benchmark e Tese** | Execução dos testes comparativos, coleta de métricas de OKR e redação final. | M8.S4 |

---

|           |S1 |S2 |S3 |S4 |S5 |S6 |S7 |S8 |S9 |S10|S11|S12|S13|S14|S15|S16|S17|S18|S19|S20|S21|S22|S23|S24|S25|S26|S27|S28|S29|S30|S31|S32|
|:--        |:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| Setup     | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Core      |   |   | x | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Polars    |   |   |   |   |   | x | x | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Spark     |   |   |   |   |   |   |   |   |   | x | x | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Métricas  |   |   |   |   |   |   |   |   |   |   |   |   |   | x | x | x | x | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Testes    |   |   | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x | x |   |   |   |   |   |   |   |   |   |   |   |   |   |
| Redação   | x |   | x |   |   | x |   |   |   | x |   |   |   | x |   |   |   |   |   | x | x | x | x | x | x | x | x | x | x | x |   |   |
| Revisão   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   | x | x | x | x |

### 7.2. Conclusão e Resultados Esperados
Ao final deste trabalho, espera-se entregar não apenas um documento acadêmico, mas uma **ferramenta de produção pronta para uso**. A conclusão do TCC deverá validar a hipótese de que é possível realizar observabilidade de alta precisão em Machine Learning com um custo de infraestrutura significativamente menor do que as soluções genéricas atuais.

Este projeto preencherá uma lacuna crítica tanto no PPC (Projeto Pedagógico de Curso) acadêmico quanto nas operações de MLOps de grande escala, posicionando o autor na fronteira entre a Ciência de Dados e a Engenharia de Plataformas.

## Referências
Engineering, Data / ML, Uber AI. **From Predictive to Generative – How Michelangelo Accelerates Uber’s AI Journey**. Blog Post. Disponível em: <https://www.uber.com/en-IN/blog/from-predictive-to-generative-ai/>. Acesso em: 25 nov. 2025

VAYYAVUR, Raj. **Why AI Projects Fail: The Importance of Strategic Alignment and Systematic Prioritization**. International Journal of Research (IJR), v. 11, n. 8, p. 386-391, 2024. Disponível em: <https://www.researchgate.net/publication/383397813_Why_AI_Projects_Fail_The_Importance_of_Strategic_Alignment_and_Systematic_Prioritization>. Acesso em: 17 dez. 2025

HAM, Tae Jun et al. **A Specialized Architecture for Object Serialization with Applications to Big Data Analytics**. In: 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2020. p. 326-339. Disponível em: <https://taejunham.github.io/data/cereal_isca2020.pdf>. Acesso em: 18 dez. 2025.
