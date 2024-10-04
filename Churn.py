# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:58:19 2024

@author: Katia Ponzetta
"""

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # procedimento Stepwise
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LogisticRegression


df_churn = pd.read_csv('customer_churn_data.csv',delimiter=',')
df_churn

#Informações sobre o tipo de dados
df_churn.info()

#Estatistica descritiva das variáveis
df_churn.describe()

# In[1.0]: Plotando Grafico para fins de analise

#Distribuição por Idade - Apresentação do valores Maximo e Minimo e Média
plt.figure(figsize=(12, 6))
sns.histplot(df_churn['Age'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribuição por Idade', fontsize=16)
plt.xlabel('Idade', fontsize=14)
plt.ylabel('Frequencia', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Distribuição por tempo de contração do serviço - Apresentação do valores Maximo e Minimo e Média
plt.figure(figsize=(12, 6))
sns.histplot(df_churn['Tenure'], bins=30, kde=True, color='lightgreen', edgecolor='black')
plt.title('Distribuição por Tempo de Contratação', fontsize=16)
plt.xlabel('Tempo de Contrato por Mês', fontsize=14)
plt.ylabel('Frequencia', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Distribuição por valor de Contrato - Apresentação do valores Maximo e Minimo e Média
plt.figure(figsize=(12, 6))
sns.histplot(df_churn['TotalCharges'], bins=30, kde=True, color='lightgreen', edgecolor='black')
plt.title('Distribuição por Valor de Contrato', fontsize=16)
plt.xlabel('Valor de Contrato', fontsize=14)
plt.ylabel('Frequencia', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Grafico de Pizza para % de Churn

churn_counts = df_churn['Churn'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['springgreen', 'magenta'])
plt.title('Distribuição de Churn (Sim e Não)', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()

#Grafico de Pizza para Qtd por Gender# Contar os valores de gênero
gender_counts = df_churn['Gender'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Distribuição de Gênero', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()

#grafico de Pizza Por tipo de Serviço
# Contar os tipos de serviço

df_churn.loc[df_churn['InternetService'].isna(), 'InternetService'] = 'Outros'

service_counts = df_churn['InternetService'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(service_counts, labels=service_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Distribuição de Tipos de Serviço de Internet', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()

#Grafico de Pizza por qtd de suporte sim e não
support_counts = df_churn['TechSupport'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Distribuição de Suporte Técnico', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()


#Grafico de Pizza por qtd de Tipo de Contrato
# Contar os tipos de contrato
contract_counts = df_churn['ContractType'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Distribuição de Tipos de Contrato', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()


# Criar uma tabela de contagem cruzada
cross_tab = pd.crosstab(df_churn['TechSupport'], df_churn['Churn'])

# Criar um gráfico de pizza para cada combinação de suporte e churn
plt.figure(figsize=(10, 8))

# Para cada tipo de suporte (Sim e Não)
for i, tech_support in enumerate(cross_tab.index):
    plt.subplot(2, 1, i + 1)  # 2 linhas, 1 coluna de gráficos
    plt.pie(cross_tab.loc[tech_support], labels=cross_tab.columns, autopct='%1.1f%%', startangle=90, 
            colors=['lightblue', 'lightcoral'])
    plt.title(f'Distribuição de Churn para Suporte Técnico: {tech_support}', fontsize=14)
    plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Ajustar o layout
plt.tight_layout()
plt.show()

# Filtrar apenas para suporte técnico "Sim"
df_tech_support_yes = df_churn[df_churn['TechSupport'] == 'Yes']

# Contar os valores de churn para suporte técnico "Sim"
churn_counts = df_tech_support_yes['Churn'].value_counts()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, 
        colors=['lightblue', 'lightcoral'])
plt.title('Distribuição de Churn para Suporte Técnico: Sim', fontsize=16)
plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Mostrar o gráfico
plt.show()


#Grafico de Pizza por tipo de serviço de Churn sim e não 
churn_counts = df_churn.groupby(['ContractType', 'Churn']).size().unstack(fill_value=0)

# Criar um gráfico de pizza para cada tipo de contrato
for contract_type in churn_counts.index:
    plt.figure(figsize=(8, 8))
    plt.pie(churn_counts.loc[contract_type], labels=churn_counts.columns, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    plt.title(f'Distribuição de Churn para {contract_type}', fontsize=16)
    plt.axis('equal')  # Para garantir que o gráfico seja um círculo
    plt.show()


# Criar uma tabela dinâmica para contar churn por tipo de contrato
churn_counts = df_churn.pivot_table(index='ContractType', columns='Churn', aggfunc='size', fill_value=0)

# Criar um gráfico de pizza para cada tipo de contrato
plt.figure(figsize=(10, 10))

# Plotar gráfico de pizza
for i, contract_type in enumerate(churn_counts.index):
    plt.subplot(2, 2, i+1)  # Ajustar o número de subgráficos conforme necessário
    plt.pie(churn_counts.loc[contract_type], labels=churn_counts.columns, autopct='%1.1f%%', startangle=90, 
            colors=['lightblue', 'lightcoral'])
    plt.title(f'Distribuição de Churn para {contract_type}', fontsize=14)
    plt.axis('equal')  # Para garantir que o gráfico seja um círculo

# Ajustar o layout
plt.tight_layout()
plt.show()


# Filtrar apenas os churns "Yes" 
df_churn_yes = df_churn[df_churn['Churn'] == 'Yes']

# Agrupar por 'Tenure' e contar os churns
qt_tenure_churn = df_churn_yes.groupby('Tenure').size().reset_index(name='Total_Churn_Yes')

qt_tenure_churn


#grafico de tendencia de tempo de contrato x churn sim e não
agg_df = df_churn.groupby(['Tenure', 'Churn'])['Tenure'].sum().unstack(fill_value=0)

# Cálculo da média
mean_values = agg_df.mean(axis=1)


# Criando o gráfico de barras
ax = agg_df.plot(kind='bar', color=['orange', 'blue'], figsize=(25, 6))

# Adicionando a linha de tendência
ax.plot(mean_values.index, mean_values, color='red', marker='o', linewidth=2, label='Média')

# Adicionando rótulos nas barras
#for container in ax.containers:
#    ax.bar_label(container)

# Configurações do gráfico
plt.xlabel('Tempo de Contrato')
plt.ylabel('Quantidade Total de Churn')
plt.title('Quantidade Total de Churn por Tempo de Contrato')
#plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['Tendencia', 'Não','Sim'])
plt.tight_layout()  # Ajusta o layout para evitar sobreposições
plt.show()
#- Verificamos que no tempo zero existem mais Churns em comparação no maiores tempos

# In[2.0]:Estimação de um modelo logístico binário pela função 'smf.glm'

#('statsmodels.formula.api') Churn em função de Idade, Tempo de Contrato e Total de Contrato

#Atribuido valores binarário para uso do modelo no stepwise
df_churn['Churn'] = df_churn['Churn'].map({'Yes': 1, 'No': 0})

modelo_churn = smf.glm(formula='Churn ~ Age + Tenure + TotalCharges', data=df_churn,
                         family=sm.families.Binomial()).fit()

# Parâmetros do 'modelo_Churn'
modelo_churn.summary()

# In[3.0]:Procedimento Stepwise, para confirmar que p_valor ao nível de significancia de 5% Fica ou sai do modelo

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/


from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise validando que o atributo 'Age' não é estatisticamente significante
step_churn = stepwise(modelo_churn, pvalue_limit=0.05)


#Atribuindo o valores de fittedvalue do Y ou seja erros para modelo de predição
df_churn['phat'] = step_churn.predict()

df_churn

#como podemos observar a variário Age explicativa não tem sigficancia estatistica sendo seu p_valor > 5%
# In[4.0]:Construção da sigmoide

# Probabilidade de Churn por 'Tempo de Contrato'    

plt.figure(figsize=(15,10))
sns.scatterplot(x=df_churn['Tenure'][df_churn['Churn'] == 0],
                y=df_churn['Churn'][df_churn['Churn'] == 0],
                color='springgreen', alpha=0.7, s=250, label='Churn = 0')
sns.scatterplot(x=df_churn['Tenure'][df_churn['Churn'] == 1],
                y=df_churn['Tenure'][df_churn['Churn'] == 1],
                color='magenta', alpha=0.7, s=250, label='Churn = 1')
sns.regplot(x=df_churn['Tenure'], y=df_churn['Churn'],
            logistic=True, ci=None, scatter=False,
            line_kws={'color': 'indigo', 'linewidth': 7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Tempo de Contrato', fontsize=20)
plt.ylabel('Probabilidade de Churn', fontsize=20)
plt.xticks(np.arange(df_churn['Tenure'].min(),
                     df_churn['Tenure'].max() + 0.01, 10),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=15)
plt.legend(fontsize=20, loc='center right')
plt.show()

#Concluimos com esse gráfico que o % de churn é maior quando o tempo de contrato é menor
#e os que estão a mais tempo custumam ser mais fieis
#Experiência Inicial Crítica: Clientes que estão no início de sua jornada com 
#o serviço podem estar insatisfeitos devido a expectativas não atendidas, 
#dificuldades de uso, ou falta de suporte. Isso indica que a experiência inicial é crucial para a retenção.
#e após o uso e conhecimento do produto aumenta, indica que entendem mais sobre o valor do serviço.

# In[5.0]:Construção de função para a definição da matriz de confusão, para definir o melhor cutoff

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores


# Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_churn['Churn'],
                predicts=df_churn['phat'],
                cutoff=0.5)

# Matriz de confusão para cutoff = 0.3 
matriz_confusao(observado=df_churn['Churn'],
                predicts=df_churn['phat'],
                cutoff=0.3)

# Matriz de confusão para cutoff = 0.7
matriz_confusao(observado=df_churn['Churn'],
                predicts=df_churn['phat'],
                cutoff=0.7)

# Matriz de confusão para cutoff = 0.8
matriz_confusao(observado=df_churn['Churn'],
                predicts=df_churn['phat'],
                cutoff=0.8)


# In[6.0]: Construção da curva ROC

from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_churn['Churn'], df_churn['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC- Apresentando um bom modelo de predição com valores de 0.807 e GINI 0.614
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()

# In[7.0]: Conclusões sobre as análises para redução de Churn

#Pontos de melhorias após analise do modelo: Para o Negocio com Foco em Redução de Churn:
#Ajustar o produto e serviço em tempo de onboarding
#e comunicação no inicio do relacionamento alinhamento de espectativa 
#Foco no Onboarding, Programa de Rentenção e Monitoramento continio de feedbacks