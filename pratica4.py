#-*- coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA

def ordenar(lista, matriz):
    valores = []
    vetores = []
    while len(lista) != 0:
        valor = max(lista)
        pos = lista.index(valor)
        vetor = matriz[pos]
        valores.append(valor)
        vetores.append(vetor)
        lista.remove(valor)
        matriz.remove(vetor)
    return  valores,vetores

# Lendo o arquivo
data = np.genfromtxt("coluna_com_label_bipolar.txt", delimiter=",")

# Total de elementos da classe Hernia
qtd_hernia = 0
for linha in data:
    if int(linha[6]) == 1:
        qtd_hernia += 1

# Classe Hernia, seprando dados
hernia = np.zeros((qtd_hernia, 6))
i = 0
while i < len(data):
    if int(data[i][6]) == 1:
        hernia[i] = data[i][0:6].tolist()
    i += 1

# Passo 1: Cálculo da média da classe.
media = np.mean(hernia, axis=0)

# Passo 2: Subtração de cada linha do vetor da matriz pela média de seu atributo, gerando uma nova matriz
matriz = np.zeros((qtd_hernia, 6))
i = 0
while i < qtd_hernia:
    matriz[i] = (hernia[i] - media)
    i += 1

# Passo 3: Matriz de covariância..
matriz_cov = np.cov(matriz, None, 0)

# Passo 4: Autovalores e Autovetores.
a_valores, a_vetores = LA.eig(matriz_cov)

# Passo 5: Ordenar de modo não crescente
valores, vetores = ordenar(list(a_valores.tolist()), list(a_vetores.tolist()))

a_valores = valores
a_vetores = vetores

# Passo 6: Construindo a matriz Q
# 6 - A: Matriz de alto vetores do maior para o menor
# Como os alto vetores já estão ordenados de acordo com
n = len(a_vetores)
Q1 = np.zeros((n, n))
i = 0
while i < len(a_vetores):
    Q1[i] = a_vetores[i]
    i += 1


# Passo 6-B
#Encontra o valor de q e criar a nova matriz

soma = 0
var = 0
total = int(0.9*len(data))
i = 0
selecionados = []
while not var >= total:
    soma += a_valores[i]
    var += soma/(len(a_vetores))
    selecionados.append(i)
    i += 1

# Pegando os autovetores selecionados
Q = Q1[0:len(selecionados)]

# Passo 7:
# Como tata-se de vetores colunas
Y = np.dot(hernia, Q.transpose())

# Passo 8:
# Calculando a nova matriz de covariância.
n_matriz_cov = np.cov(Y, None, 0)

print "Matriz de Covariância Original: \n", matriz_cov , "\n"
print "Nova Matriz de Covariância: \n", n_matriz_cov , "\n"


