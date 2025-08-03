import pandas as pd
import os

file_path = os.path.join('notebook', 'data', 'stud.csv')
print(file_path)
df = pd.read_csv(file_path)