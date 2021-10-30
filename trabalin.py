import numpy as np
import pandas as pd
import math
import copy
import time

class Nodo(object):
    def __init__(self):
        self.pai = None
        self.filhos = None
        self.atributo = None
        self.ganho = None

# Retorna a entropia da classe (info)
# data: dataframe
# classe: nome da classe preditiva
# lista_valores_unicos: lista com os valores unicos da classe preditiva
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
# classe: nome da classe preditiva
# lista_valores_unicos: lista com os valores únicos da classe preditiva
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
# classe: nome da classe preditiva
# lista_valores_unicos: lista com os valores únicos da classe preditiva
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
    amostra = lista_atributos.columns.values.tolist()

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

    # Se é uma classe pura retorna como nodo folha
    if mesma_classe(data, classe, lista_valores_unicos):
        nodo.atributo = data[classe].iloc[0]
        nodo.pai = valor_pai
        return nodo

    # Se o dataframe for vazio retorna nodo folha com a classe mais frequente
    if data.empty:
        nodo.atributo = classe_mais_frequente(data, classe)
        nodo.pai = valor_pai
        return nodo
    
    # Seleciona a partir de uma amostra m o atributo com maior ganho
    amostra = get_amostra(data, classe, num_atributos)
    atributo, ganho_max = maior_ganho(data, amostra, classe, lista_valores_unicos)

    nodo.atributo = atributo
    nodo.ganho = ganho_max
    nodo.filhos = []

    # Busca os valores únicos do atributo
    valores_unicos = data[atributo].unique()

    if data[atributo].dtype.kind in 'biufc':
        criterio = data[atributo].mean()
        maiores = data[data[atributo] > criterio]
        menores = data[data[atributo] <= criterio]

        valores_unicos = ["<= "+str(criterio) , "> "+str(criterio)]
        for valor in valores_unicos:
            if (valor[0] == "<"):
                lista_valores_unicos = menores[classe].unique()
                menores = menores.reset_index(drop=True)
                novo_nodo = arvore_decisao(menores, classe, lista_valores_unicos, num_atributos, valor)
            else:
                lista_valores_unicos = maiores[classe].unique()
                maiores = maiores.reset_index(drop=True)
                novo_nodo = arvore_decisao(maiores, classe, lista_valores_unicos, num_atributos, valor)
            
            novo_nodo.pai = valor
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
            print(tab*"\t" + "----- " + str(root.pai) + " (ganho: " + str(0 if (root.ganho is None) else round(root.ganho,2)) + ") --- " + str(root.atributo))
        else:
            print("\t" + str(root.atributo))
        if (root.filhos):
            tab += 1
            for child in root.filhos:
                printTree(child,tab)
            print()

# Classifica uma instancia
# instane: row com uma instancia a ser classificada
# lista_atributos: lista de atributos sem a classe preditiva
# nodo: nodo raiz
def classify(instance, nodo):
    resultado = None
    if nodo:
        if not nodo.filhos:
            return nodo.atributo

        # Se o atributo é do tipo numérico
        if instance[nodo.atributo].dtype.kind in 'biufc':
            for filho in nodo.filhos:
                if "<=" in filho.pai:
                    valor = float(filho.pai.replace("<= ", ""))
                    if float(instance.iloc[0][nodo.atributo]) <= valor:
                        if not filho.filhos:
                            return filho.atributo
                        else:
                            resultado = classify(instance, filho)
                else:
                    valor = float(filho.pai.replace("> ", ""))
                    if float(instance.iloc[0][nodo.atributo]) > valor:
                        if not filho.filhos:
                            return filho.atributo
                        else:
                            resultado = classify(instance, filho)
        else:
            for filho in nodo.filhos:
                if filho.pai == instance.iloc[0][nodo.atributo]:
                    if not filho.filhos:
                        return filho.atributo
                    else:
                        resultado = classify(instance, filho)

    return resultado

# Retorna subconjunto de dataFrame a partir de bagging
# data: dataframe
def bootstrap(data):
    dataset = data.copy(deep=True)
    length = len(data)
    tamanho = round(length * 0.632)
    randlist = data.sample(n = tamanho)
    #dataset.merge(randlist, left_index=True, right_index=True, how='right')
    return randlist

# Retorna um ensemble de árvores
def geraEnsemble(data, num_arvores, classe, lista_valores_unicos, num_atributos):
    ensemble = []

    for x in range(0,num_arvores):
        dataset = bootstrap(data)
        lista_valores_unicos = dataset[classe].unique()
        tree = arvore_decisao(dataset, classe, lista_valores_unicos, num_atributos, 'raiz')
        ensemble.append(tree)

    return ensemble

# Retorna lista de subgrupos de treinamento e teste para validação cruzada
# dataframe: dataframe
# kFold: numero de subgrupos
def crossValidation(data, kFold):
    crossValidation = []
    dataframe = data.sample(frac = 1)
    step = len(dataframe) / kFold
    for x in range(kFold):
        first = int(x * step)
        last = int(((x + 1) * step) - 1)
        crossValidation.append(dataframe[first:last].reset_index(drop=True))
    return crossValidation

def valida(data, num_atributos, num_arvores, classe, lista_valores_unicos, k_folds):
    corretos = 0
    errados = 0
    predicoes = []
    dados = crossValidation(data, k_folds)
    for x in range(len(dados)):
        dados_copia = copy.deepcopy(dados)
        teste = dados_copia.pop(x)
        treino = pd.concat(dados_copia)

        nodos = geraEnsemble(treino, num_arvores, classe, lista_valores_unicos, num_atributos)

        for index, linha in teste.iterrows():
            original = linha[classe]
            instance = teste.loc[[index]]
            for nodo in nodos:
                resultado = classify(instance, nodo)
                predicoes.append(resultado)
            predito = max(set(predicoes), key=predicoes.count) 
            if predito == original:
                corretos += 1
            else:
                errados += 1
    
    print("Total: " + str(corretos + errados))
    print("Corretos: " + str(corretos) + " ou " + "{:.2f}".format( (corretos/(corretos+errados)) * 100 ) + " %")
    print("Errados: " + str(errados) + " ou " + "{:.2f}".format( (errados/(corretos+errados)) * 100 ) + " %")


# Dataset a ser analisado
data = pd.read_csv('benchmark.csv', sep=';')
classe = "Joga"

# data = pd.read_csv('house-votes-84.tsv', sep='\t')
# classe = "target"

# data = pd.read_csv('wine-recognition.tsv', sep='\t')
# classe = "target"

lista_valores_unicos = data[classe].unique()

# Número de atributos da amostragem = raiz quadrada do numero de colunas
num_atributos = round(math.sqrt(data.shape[1]))

# Número de arvores no ensemble
num_arvores = [1,10,25,50]

# Número de folds
k_folds = 10

start = time.time()
for numero in num_arvores:
    valida(data, num_atributos, numero, classe, lista_valores_unicos, k_folds)
end = time.time()
print(end - start)

# root = arvore_decisao(data, classe, lista_valores_unicos, num_atributos, 'raiz')
# printTree(root,0)

