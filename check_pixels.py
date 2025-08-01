import pandas as pd

# Load the dataset
df = pd.read_csv('D:/Downloads/MOD09GA_Ren_2000-01-01_to_2019-12-31_new5 - MOD09GA_Ren_2000-01-01_to_2019-12-31_new5.csv')

# Aggregate by pixel_id
pixel_summary = df.groupby('pixel_id').agg({
    'glacier_fraction': 'mean', 
    'edge_effect': lambda x: x.mode().iloc[0],
    'longitude': 'first',
    'latitude': 'first'
}).reset_index()

pixel_summary.columns = ['pixel_id', 'glacier_fraction', 'edge_effect', 'longitude', 'latitude']

print('Pixel Summary:')
print('=' * 70)
print(f'{"Pixel ID":>10} | {"Glacier Fraction":>15} | {"Edge Effect":>11} | {"Longitude":>10} | {"Latitude":>9}')
print('-' * 70)

for _, row in pixel_summary.iterrows():
    print(f'{row["pixel_id"]:>10} | {row["glacier_fraction"]:>15.3f} | {row["edge_effect"]:>11} | {row["longitude"]:>10.4f} | {row["latitude"]:>9.4f}')

print(f'\nTotal pixels: {len(pixel_summary)}')