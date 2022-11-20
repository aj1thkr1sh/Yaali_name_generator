from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1

def findFiles(path):
    return glob.glob(path)

def unicodeToASCII(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    with open(filename, encoding="utf-8") as file:
        return [unicodeToASCII(line.strip()) for line in file]


category_lines = {}
all_categories = []
for filename in findFiles('g_data/names/*.txt'):
    print(filename)
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found')

print("# categories: ", n_categories, all_categories)
# print(unicodeToASCII("O'Néàl"))