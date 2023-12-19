import pandas as pd
import pickle

# Baca data dari file Excel
csv_file_path = 'heart.csv'
df = pd.read_csv(csv_file_path)

# Simpan DataFrame ke file pickle
pickle_file_path = 'data.pickle'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(df, pickle_file)

print(f'DataFrame telah disimpan dalam file pickle: {pickle_file_path}')
