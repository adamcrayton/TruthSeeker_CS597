#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef, precision_score, recall_score
from tabulate import tabulate
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="eval_matrix_config")

def process_evalmatrix(cfg : DictConfig):
    
    df_matrix = pd.read_csv(cfg['prediction_csv'])


    # Evaluation Metrics


    # Assuming y_true contains the true labels and y_pred contains the predicted labels
    y_true = df_matrix['Labels']
    y_pred = df_matrix['Prediction']

    acc = accuracy_score(y_true, y_pred)
    cohen = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Cohen's Kappa: {cohen}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Precision: {prec}")
    print(f"Recall: {recall}")
    
    table = [
    ["Accuracy", acc],
    ["Cohen's Kappa", cohen],
    ["F1 Score", f1],
    ["Matthews Correlation Coefficient", mcc],
    ["Precision", prec],
    ["Recall", recall]
]

   # Print the transposed table
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign="right"))

if __name__ == '__main__':
    process_evalmatrix()


