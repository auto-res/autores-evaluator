import pandas as pd
import seaborn as sns


def retrieve_titanicdata():
    titanic = sns.load_dataset('titanic')
    titanic.dropna(inplace=True)

    categorical_columns = ['sex', 'class', 'embarked', 'who', 'deck', 'embark_town', 'alive', 'alone']
    titanic_encoded = pd.get_dummies(titanic, columns=categorical_columns)

    return titanic_encoded
