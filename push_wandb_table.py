import pandas as pd
import wandb

df = pd.read_csv('/Users/akashpatel/Tokyo.csv')
run = wandb.init(project="PeopleFlow", job_type="Logging table")

table = wandb.Table(data=df)
run.log({"table_key": table})