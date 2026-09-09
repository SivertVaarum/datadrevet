import pandas

document = pandas.read_csv("../smoking_driking_dataset_Ver01.csv")

def numericalExplore(column):
    print(column + " ######################################################")
    column = document[column]
    print(column.describe())
    print("not-null: ")
    print(column.notna())

numericalExplore("age")
numericalExplore("height")
numericalExplore("weight")
numericalExplore("waistline")
numericalExplore("sight_left")
numericalExplore("sight_right")
numericalExplore("hear_left")
numericalExplore("hear_right")
numericalExplore("SBP")
numericalExplore("DBP")
numericalExplore("BLDS")
numericalExplore("tot_chole")





