import pandas as pd
from sklearn.linear_model import LogisticRegression


def read_csvs(prediction_csvs):
    gt = None
    predictions = []
    columns = []
    final_features = pd.DataFrame()
    for i, prediction_csv in enumerate(prediction_csvs):
        df = pd.read_csv(prediction_csv)
        gt = df['label']
        index = df['index']
        final_features['prediction_' + str(i)] = df['prediction']
        predictions.append(df['prediction'])
        columns.append('prediction_'+str(i))
    
    return final_features.values, gt.values, index.values

def train_ensemble_stacking(train_prediction_csvs, test_prediction_csvs, dest_path):
    train_stackedX, train_y, _ = read_csvs(train_prediction_csvs)
    test_stackedX, test_y, test_index = read_csvs(test_prediction_csvs)
    model = LogisticRegression()
    model.fit(train_stackedX, train_y)
    dest_df = pd.DataFrame()
    dest_df['index'] = test_index
    dest_df['prediction'] = model.predict(test_stackedX)
    dest_df['label'] = test_y
    dest_df.to_csv(dest_path, index=False)

train_ensemble_stacking(train_prediction_csvs=
    [
        '/home/student/workspace/Truthseeker/prediction1.csv',
        '/home/student/workspace/Truthseeker/prediction2.csv'
     ],test_prediction_csvs=
      [
        '/home/student/workspace/Truthseeker/prediction1.csv',
        '/home/student/workspace/Truthseeker/prediction2.csv'
     ],
     dest_path='/home/student/workspace/Truthseeker/ensemble_prediction.csv'
)