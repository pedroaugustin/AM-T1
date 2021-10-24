import numpy as np
import pandas as pd
from collections import Counter

class Nodo(object):
    def __init__(self):
        self.pai = None
        self.filhos = None
        self.atributo = None
        self.ganho = None

# Retorna a entropia da classe (info)
# data: dataframe
# classe: nome da classe
# lista_valores_unicos: lista com os valores unicos da classe
def entropia_classe(data, classe, lista_valores_unicos):
    entropia_classe = 0
    num_linhas = data.shape[0]
    
    # Para cada valor único da classe 
    for valor in lista_valores_unicos: 
        # Número de linhas da classe que contem o valor único da classe
        num_linhas_classe = data[data[classe] == valor].shape[0] 
        entropia_total = - (num_linhas_classe/num_linhas)*np.log2(num_linhas_classe/num_linhas)
        # Entropia da classe
        entropia_classe += entropia_total 
    
    return entropia_classe

# Retorna a entropia de um atributo (infoA)
# atributo_data: dataframe com os valores únicos de um atributo
# classe: nome da classe
# lista_valores_unicos: lista com os valores únicos da classe
def entropia_valor(atributo_data, classe, lista_valores_unicos):
    entropia = 0
    num_linhas = atributo_data.shape[0]

    for valor in lista_valores_unicos:
        entropia_valor = 0
        # Número de linhas do dataframe do atributo com os valores únicos da classe
        num_linhas_classe = atributo_data[atributo_data[classe] == valor].shape[0]
        
        if num_linhas_classe != 0:
            # Probabilidade do valor único
            valor_probabilidade = num_linhas_classe/num_linhas
            
            # Entropia do valor único
            entropia_valor = - valor_probabilidade * np.log2(valor_probabilidade) 

        entropia += entropia_valor

    return entropia

# Retorna o ganho de informação de um atributo
# atributo: nome do atributo que queremos o ganho de informação
# data: dataframe 
# classe: nome da classe
# lista_valores_unicos: lista com os valores únicos da classe
def ganho(atributo, data, classe, lista_valores_unicos):
    num_linhas = data.shape[0]
    info_atributo = 0.0
    # Valores únicos de um atributo
    atributos_unicos = data[atributo].unique()
    
    for atributo_unico in atributos_unicos:
        # Linhas que tem o mesmo valor do valor único do atributo
        atributo_data = data[data[atributo] == atributo_unico]
        atributo_count = atributo_data.shape[0]

        # Calcula entropia para o valor único do atributo
        atributo_entropia = entropia_valor(atributo_data, classe, lista_valores_unicos)
        atributo_probabilidade = atributo_count/num_linhas

        # Calcula a info do valor único do atributo
        info_atributo += atributo_probabilidade * atributo_entropia 

    # Calcula a entropia total info - info(atributo)
    ganho = entropia_classe(data, classe, lista_valores_unicos) - info_atributo 
    
    return ganho

# Retorna uma amostragem do dataframe
# data: dataframe
# classe: nome da classe
# num_atributos: numero de atributos a serem selecionados
def get_amostra(data, classe, num_atributos):
    lista_atributos = data.drop(columns = classe)
    amostra = lista_atributos.sample(n=num_atributos,axis='columns')

    return amostra

# Retorna o atributo com maior ganho
# data: dataframe com atributos
# classe: nome da classe
# lista_valores_unicos: lista de valores unicos da classe
def maior_ganho(data, amostra, classe, lista_valores_unicos):
    melhor_atributo = None
    ganho_max = -1
      
    for atributo in amostra:  
        ganho_atributo = ganho(atributo, data, classe, lista_valores_unicos)
        if ganho_max < ganho_atributo: 
            ganho_max = ganho_atributo
            melhor_atributo = atributo
            
    return melhor_atributo, ganho_max

# Retorna 1 se é uma classe pura e 0 se não é 
# atributo_data: dataframe
# classe: classe
# lista_valores_unicos: lista de valores unicos da classe
def mesma_classe(atributo_data, classe, lista_valores_unicos):
    if entropia_valor(atributo_data, classe, lista_valores_unicos) == 0:
        return 1
    return 0

# Retorna a classe mais frequente 
# data: dataframe
# classe: classe
def classe_mais_frequente(data, classe):
    return data[classe].value_counts().idxmax()

# Cria a arvore de decisao
# data: dataframe
# classe: classe a ser predizida
# lista_valores_unicos: lista de valores unicos da classe
# num_atributos: numero de atributos da amostragem
# valor_pai: valor do nodo pai
def arvore_decisao(data,  classe, lista_valores_unicos, num_atributos, valor_pai):
    nodo = Nodo()

    if mesma_classe(data, classe, lista_valores_unicos):
        nodo.atributo = data[classe].iloc[0]
        nodo.pai = valor_pai
        return nodo

    if data.empty:
        nodo.atributo = classe_mais_frequente(data, classe)
        nodo.pai = valor_pai
        return nodo
    
    amostra = get_amostra(data, classe, num_atributos)
    atributo, ganho_max = maior_ganho(data, amostra, classe, lista_valores_unicos)

    nodo.atributo = atributo
    nodo.ganho = ganho_max
    nodo.filhos = []

    valores_unicos = data[atributo].unique()

    if data[atributo].dtype.kind in 'biufc':
        criterio = data[atributo].mean()
        maiores = data[data[atributo] > criterio]
        menores = data[data[atributo] <= criterio]
        for valor in valores_unicos:
            # Linhas que tem o mesmo valor do valor único do atributo
            if valor > criterio:
                atributo_data = maiores[maiores[atributo] == valor]
            elif valor <= criterio:
                atributo_data = menores[menores[atributo] == valor]
            novo_nodo = arvore_decisao(atributo_data, classe, lista_valores_unicos, num_atributos)
            nodo.filhos.append(novo_nodo)
    else:
        for valor in valores_unicos:
            nova_data = data[data[atributo] == valor]

            novo_nodo = arvore_decisao(nova_data, classe, lista_valores_unicos, num_atributos, valor)
            novo_nodo.pai = valor
            nodo.filhos.append(novo_nodo)

    return nodo

# Impressão da árvore
def printTree(root,tab):
    if root:
        if root.pai:
            print(tab*"\t" + "----- " + root.pai + " --- " + root.atributo)
        else:
            print("-----  " + root.atributo + "  -----")
        if (root.filhos):
            tab += 1
            for child in root.filhos:
                printTree(child,tab)
            print()


data = pd.read_csv('benchmark.csv', sep=';')

class_list = data['Joga'].unique()
num_atributos = 3
entropy = ganho('Tempo', data, 'Joga', class_list)
root = arvore_decisao(data, 'Joga', class_list, num_atributos, 'raiz')

printTree(root,0)

