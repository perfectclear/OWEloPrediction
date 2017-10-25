import pandas as pd
import httplib2
from time import sleep
from namesandstuff import nouns

def scrapename(name):
    conn = httplib2.Http()
    resp, content = conn.request(uri = 'https://www.overbuff.com/search?q={}'.format(name))
    content = str(content)
    content = content.split('<a href="')
    data = []
    for i in content:
        if i.startswith('/players/pc/'):
            data.append(i)

    names = []
    data = pd.Series(data)
    if data.size != 0:
        names.append(data.str.extract('([a-zA-Z0-9@*#-/-\-]+)', expand=False))
        for i in names:
            names.remove(i)
            i = str(i).split('/')
            names.append(i)

        actualnames = []
        for i in names[0]:
            if '-' in str(i) and '\n' not in str(i):
                actualnames.append(str(i))
        actualnames = set(actualnames)
    else:
        actualnames = set()
    return actualnames

with open('nameslist.txt','r') as f:
       content = f.readlines()
       for line in content:
           names = line.split(",")
names = set(names)
names.remove('')
# print(names)
i=0
while len(names) < 10000 and i < 1540:
    noun = nouns[i]
    newnames = scrapename(noun)
    with open("nameslist.txt", 'a') as f:
        for name in newnames:
            f.write(name + ",")
    names |= newnames
    # print(len(names))
    i+=1
    sleep(20)
