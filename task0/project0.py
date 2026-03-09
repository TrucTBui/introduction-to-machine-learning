import pandas as pd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_features = train_df.iloc[:, 2:] 
calculated_mean = train_features.mean(axis=1)

test_features = test_df.iloc[:, 1:] 
test_predictions = test_features.mean(axis=1)

results = pd.DataFrame({
    'Id': test_df['Id'],
    'y': test_predictions
})

results.to_csv('submission.csv', index=False)