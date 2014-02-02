import sys

with open(sys.argv[1], 'r') as f:
    with open(sys.argv[2], 'w') as g:
        for i,line in enumerate(f):
            if i == 0:
                continue
            g.write(f + '\n')
