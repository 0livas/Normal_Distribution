import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot, shapiro

# database
df = pd.read_csv("Database/Sorting_Algorithm.csv")

# Obter algoritmos únicos
algoritmos = df['algorithm'].unique()


results = {}

# Loop pelos algoritmos
for algoritmo in algoritmos:
    dados = df[df['algorithm'] == algoritmo]['execution_time_ms']
    

    media = dados.mean()
    desvio = dados.std()
    
    # Histograma c/curva normal
    plt.figure(figsize=(10, 4))
    sns.histplot(dados, kde=True, stat="density", bins=20, label='Dados')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, media, desvio)
    plt.plot(x, p, 'k', linewidth=2, label='Normal teórica')
    plt.title(f'{algoritmo} - Histograma com Curva Normal')
    plt.xlabel('Tempo de Execução (ms)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Gráfico Q-Q
    plt.figure(figsize=(5, 5))
    probplot(dados, dist="norm", plot=plt)
    plt.title(f'{algoritmo} - Gráfico Q-Q')
    plt.grid(True)
    plt.show()
    
    #resultados
    results[algoritmo] = {
        'média': media,
        'desvio padrão': desvio
    }

# Exibir resultados finais
print("Estatísticas descritivas por algoritmo:\n")
for alg, res in results.items():
    print(f"{alg}: Média = {res['média']:.4f} ms, Desvio Padrão = {res['desvio padrão']:.4f} ms")
