import pandas

doc = pandas.read_csv("data_cleaned.csv")

def encodeSex():
    global doc
    doc['sex_female'] = (doc['sex'] == 'Female').astype(int)
    doc = doc.drop(columns=["sex"])

def encodeDrink():
    global doc
    doc['DRINK'] = (doc["DRK_YN"] == "Y").astype(int)
    doc = doc.drop(columns=["DRK_YN"])

def save():
    doc.to_csv("transformed.csv")

encodeSex()
encodeDrink()
save()


