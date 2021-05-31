file = ''
with open("bank.csv") as f: 
    phrase = ','.join(f.read().split(";"))
    file+=phrase
print(file)
with open("banks.csv","w") as f:
    f.write(file)
    f.close()