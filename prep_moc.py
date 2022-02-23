import pandas as pd

moc_df = pd.read_csv("congress_members.csv")

def frac(dataframe, fraction, other_info=None):
    """Returns fraction of data"""
    return dataframe.sample(frac=fraction)

training_data = frac(moc_df, 0.7)
training_data.sort_values(by = ['moc_id'], inplace = True)

test_data = pd.merge(moc_df, training_data, on=['moc_id', 'state_district', 'chamber', 'name', 'party', 'twitter_handle'], how="outer", indicator=True
              ).query('_merge=="left_only"')

print(training_data)
print(test_data)

training_data.to_csv("moc_training_data.csv")
test_data.to_csv("moc_test_data.csv")