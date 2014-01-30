import sys

f1,f2 = map(file, sys.argv[1:])
f1.readline()
f2.readline()

for i,j in zip(f1, f2):
    for k,l in zip(i.split(','), j.split(',')):
        try:
            if float(k) != float(l):
                print k, l
        except:
                print k, l
