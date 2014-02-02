import sys
import math

# First pass over the data: average all columns.
# Second pass: replace each NA with the average for that column

totals = [0 for i in xrange(800)]
counts = [0 for i in xrange(800)]

classifying = False
if sys.argv[2] == 'c':
    classifying = True

def thresh(x):
    if x > 0.5:
        return 1
    else:
        return 0

lines = 0
print 'Munging on ' + sys.argv[1]
for line in open(sys.argv[1], 'r'):
    if lines == 0:
        lines += 1
        continue
    lines +=1

    for i,c in enumerate(line.split(',')[1:]):
        try:
            totals[i] += math.log(abs(float(c) + 0.001))
            counts[i] += 1
        except:
            pass

    if lines % 10000 == 0:
        print lines, 'completed for first pass'

avgs = [totals[i]/counts[i] if counts[i]>0 else 0 for i in xrange(len(totals))]
print 'Averages:', avgs


lines = 0
if classifying:
    name = sys.argv[1] + '_cleaned_c.csv'
else:
    name = sys.argv[1] + '_cleaned_c.csv'
with open(name, 'w') as f:
    label_i = 0
    for line in open(sys.argv[1], 'r'):
        if lines == 0:
            lines += 1
            continue
        if label_i == 0:
            label_i = len(line.split(','))-2
        f.write(str(lines))
        lines += 1

        for i,c in enumerate(line.split(',')[1:]):
            f.write(',')
            try:
                if i == label_i:
                    if classifying:
                        f.write("%f" % thresh(float(c)))
                    else:
                        f.write("%f" % float(c))
                else:
                    f.write("%f" % math.log(abs(float(c))))
            except:
                f.write("%f" % avgs[i])
        f.write('\n')
        
        if lines % 10000 == 0:
            print lines, 'completed for second pass'
