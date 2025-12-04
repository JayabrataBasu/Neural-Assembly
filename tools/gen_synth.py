#!/usr/bin/env python3
"""
Generate synthetic CSV datasets for the Neural Assembly project.
Usage: python3 tools/gen_synth.py <train_rows> <test_rows> <features> <classes>
Example: python3/tools/gen_synth.py 2000 500 8 3
"""
import sys, random

def gen_csv(path, n_rows, n_cols):
    with open(path, 'w') as f:
        for i in range(n_rows):
            row = [f"{random.random():.6f}" for _ in range(n_cols)]
            f.write(','.join(row) + '\n')

def gen_labels(path, n_rows, n_classes):
    with open(path, 'w') as f:
        for i in range(n_rows):
            label = random.randrange(n_classes)
            f.write(str(label) + '\n')

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: gen_synth.py <train_rows> <test_rows> <features> <classes>')
        sys.exit(1)
    train_rows = int(sys.argv[1])
    test_rows = int(sys.argv[2])
    n_features = int(sys.argv[3])
    n_classes = int(sys.argv[4])
    print(f'Generating train {train_rows} rows, test {test_rows} rows, features={n_features}, classes={n_classes}')
    gen_csv('csv/train.csv', train_rows, n_features)
    gen_labels('csv/train_labels.csv', train_rows, n_classes)
    gen_csv('csv/test.csv', test_rows, n_features)
    gen_labels('csv/test_labels.csv', test_rows, n_classes)
    print('Done')
