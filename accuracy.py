import pandas as pd


df = pd.read_csv('your_file.csv')


accuracy = (df['is_duplicate'] == df['dup']).mean()
print(f'Accuracy: {accuracy * 100:.2f}%')