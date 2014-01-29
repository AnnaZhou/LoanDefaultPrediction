import sys

# First pass over the data: average all columns.
# Second pass: replace each NA with the average for that column

totals = [0 for i in xrange(800)]
counts = [0 for i in xrange(800)]

lines = 0
print 'Munging on ' + sys.argv[1]
for line in open(sys.argv[1], 'r'):
    if lines == 0:
        lines += 1
        continue
    lines +=1

    for i,c in enumerate(line.split(',')[1:]):
        try:
            totals[i] += float(c)
            counts[i] += 1
        except:
            pass

    if lines % 10000 == 0:
        print lines, 'completed for first pass'

avgs = [totals[i]/counts[i] if counts[i]>0 else 0 for i in xrange(len(totals))]
print 'Averages:', avgs

lines = 0
with open(sys.argv[1]+'_cleaned.csv', 'w') as f:
    for line in open(sys.argv[1], 'r'):
        if lines == 0:
            f.write(line)
            lines += 1
            continue
        f.write(str(lines))
        lines += 1

        for i,c in enumerate(line.split(',')[1:]):
            f.write(',')
            try:
                f.write(str(float(c)))
            except:
                f.write(str(avgs[i]))
        f.write('\n')
        
        if lines % 10000 == 0:
            print lines, 'completed for second pass'
