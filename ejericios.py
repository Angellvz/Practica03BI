import pandas as pd
import numpy as np

# Cargar el archivo u.data
data = pd.read_csv("ml-100k/ml-100k/u.data", sep='\t', names=["userID", "itemId", "rating", "timestamp"])

# Supongamos que deseas obtener las calificaciones de los usuarios 1 y 2
usuario1_ratings = data[data['userID'] == 1]['rating'].tolist()
usuario2_ratings = data[data['userID'] == 2]['rating'].tolist()

# Asegurarse de que ambas listas tengan la misma longitud rellenando con ceros
max_length = max(len(usuario1_ratings), len(usuario2_ratings))
usuario1_ratings += [0] * (max_length - len(usuario1_ratings))
usuario2_ratings += [0] * (max_length - len(usuario2_ratings))

# Calcular la correlación de Pearson
correlation = np.corrcoef(usuario1_ratings, usuario2_ratings)[0, 1]
print("Correlación de Pearson entre Usuario 1 y Usuario 2:", correlation)
