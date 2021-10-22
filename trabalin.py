import math
import numpy as np
import pandas as pd
from collections import Counter

class Node:
    # Classe que implementa um nodo
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value

class Tree:
    # Classe que implementa a árvore de decisão
    def __init__(self, data):
        self.data = data

    # Calcula a entropia da classe
    def entropia (self, df, classe):
        valores = df[classe].tolist()
        valores_diferentes = df[classe].unique()
        total_valores = len(valores)
        entropy = 0

        for val in valores_diferentes:
            entropy=entropy-(valores.count(val)/total_valores)*math.log(valores.count(val)/total_valores,2)

        return entropy
    
    # Calcula a entropia baseado na divisao dos dados em um atributo A
    def entropia_A (self, df, atributo, classe):
        valores = df[atributo].tolist()
        total_valores = len(valores)
        count = 0

        valores_diferentes = df[atributo].value_counts().keys().tolist()
        counts_diferentes = df[atributo].value_counts().tolist()
        
        print(valores_diferentes)
        print(counts_diferentes)
        
        entropy = 0

        for val in valores_diferentes:
            print(val)
            entropy=entropy+(counts_diferentes[count]/total_valores)*self.entropia(df,atributo,classe)

        return entropy

df = pd.read_csv('benchmark.csv', sep=';')
tree = Tree(df)
entropia = tree.entropia_A(df,'Joga')
#print(df.values.tolist())
print(entropia)
