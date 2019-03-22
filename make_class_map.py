#!/usr/bin/env python
"""Class map generator.

Generates a class mapping file that lets us map between class indices and class
names.  Reads standard input and writes to standard output. To generate a usable
class map, pass in the CSV for the entire training set so that we see all
possible labels.

Input: a CSV file with a single header line and matching data lines, e.g.,
  fname,aso_ids,label
  196702,"/m/06_fw","Skateboard"
  a3bbf36a7fff3e63bcf6d2e250bd79e8,"/m/07s0dtb,/m/07plz5l","Gasp,Sigh"

Output: a CSV file with data lines of the form class_index,class_name with one
entry per unique class in the input, and 0-based class indices assigned in
ascending order of class name, e.g.,
  0,Cough
"""

import csv
import sys

unique_labels = set()
reader = csv.reader(sys.stdin)
reader.next()  # Skip header
for row in reader:
  unique_labels.update(row[2].split(','))
unique_labels = sorted(unique_labels)

csv_writer = csv.DictWriter(sys.stdout, fieldnames=['class_index', 'class_name'])
for (class_index, class_name) in enumerate(unique_labels):
  csv_writer.writerow({'class_index': class_index, 'class_name':class_name})
