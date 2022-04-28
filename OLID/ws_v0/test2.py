import pandas as pd
import re

df = pd.read_csv("hwz.csv")

df['comment'] = df['comment'].fillna("")

def remove_un(un):
    return re.sub('^(.*?said: )','', un)
df['comment'] = df.comment.apply(remove_un)


df.to_csv('bionix2removeun.csv')