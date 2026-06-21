import csv
p='eval_output/mispredictions_summary.csv'
found = False
with open(p,encoding='utf-8') as f:
    r=csv.DictReader(f)
    for row in r:
        if row['true'].startswith('Mango') and row['pred'].startswith('Banana'):
            print(row['image'])
            found = True
            break
if not found:
    print('NONE')
