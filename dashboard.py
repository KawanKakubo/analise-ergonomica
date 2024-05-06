import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados do CSV
posture_data = pd.read_csv("posture_data.csv")

# Conta quantos registros de cada status temos
status_counts = posture_data['status'].value_counts()

# Cria um gráfico de barras para visualizar o tempo em cada status
plt.bar(status_counts.index, status_counts.values)
plt.xlabel("Status")
plt.ylabel("Contagem")
plt.title("Tempo gasto em cada status de postura")
plt.show()
