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
    
    # Teste de Shapiro-Wilk
    stat, p_valor = shapiro(dados)
    
    #resultados
    results[algoritmo] = {
        'W': stat,
        'p-valor': p_valor,
        'média': media,
        'desvio padrão': desvio
    }

# Exibir resultados finais
print("Resultados do Teste de Shapiro-Wilk + Estatísticas Descritivas:\n")
for alg, res in results.items():
    interpretacao = "✅ Provavelmente normal" if res['p-valor'] > 0.05 else "❌ Provavelmente não normal"
    print(f"{alg}: W = {res['W']:.4f}, p = {res['p-valor']:.5f} → {interpretacao}")
    print(f"     Média = {res['média']:.4f} ms, Desvio Padrão = {res['desvio padrão']:.4f} ms\n")
