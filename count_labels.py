import csv
from collections import Counter
p = 'data/dataset_realistic_bug.csv'
with open(p, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cnt = Counter()
    rows = 0
    for r in reader:
        rows += 1
        cnt[r['label']] += 1
print('ROWS:', rows)
print('LABEL COUNTS:')
for k,v in cnt.items():
    print(k, v)
