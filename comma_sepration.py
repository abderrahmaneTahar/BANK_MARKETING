file = ''
with open("../bank-full.csv") as f: 
    phrase = ','.join(f.read().split(";"))
    file+=phrase
print(file)
with open("../banks-full.csv","w") as f:
    f.write(file)
    f.close()