import csv
from collections import Counter, defaultdict
import os
p=os.path.join(os.path.dirname(__file__), '..','eval_output','mispredictions_summary.csv')
if not os.path.exists(p):
    print('CSV not found:',p); raise SystemExit(1)
conf=Counter()
true_counts=Counter()
pred_counts=Counter()
with open(p,newline='',encoding='utf-8') as f:
    reader=csv.DictReader(f)
    for r in reader:
        t=r['true']; q=r['pred']
        conf[(t,q)]+=1
        true_counts[t]+=1
        pred_counts[q]+=1
print('\nTop 12 confusion pairs (true -> pred):')
for (t,q),c in conf.most_common(12):
    print(f"{c:4d}  {t} -> {q}")
print('\nTop 12 true classes with most mispredictions:')
for t,c in true_counts.most_common(12):
    print(f"{c:4d}  {t}")
print('\nTop 12 predicted classes (as mistaken outputs):')
for q,c in pred_counts.most_common(12):
    print(f"{c:4d}  {q}")
