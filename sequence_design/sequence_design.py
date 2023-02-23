'''
Dummy python file for creating sequences. Sequence designing is missing, only creates random sequences.
Used for testing command arguments.
'''
#import RNA
#import infrared as ir
#import infrared.rna as rna
import random
import math
from collections import Counter
import time
import argparse

parser = argparse.ArgumentParser(description='A program for creating new sequences from a fasta file using Infrared.')
parser.add_argument('-i', '--input', type=str, help='the input fasta file')
parser.add_argument('-o', '--output', type=str, help='the output file for the created sequences')
args = parser.parse_args()


if args.input and args.output:
    input_file = args.input
    output_file = args.output

    structures = []
    sequences = []
    try:
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                if i%3==2:
                    structures.append(line[:-1])
                elif i%3==1:
                    sequences.append(line[:-1])
    except:
        print("Fasta file could not be read.")
        print("Fasta file must be in the format:")
        print("\t - Name")
        print("\t - sequence")
        print("\t - structure")
        exit()

    new_sequences = []

    bases = ["A", "C", "G", "U"]
    for target, seq in zip(structures, sequences):
        new_seq = random.choices(bases, k = len(seq))
        new_sequences.append("".join(new_seq))

    with open(output_file, "w") as f:
        n = len(new_sequences) - 1
        for i, seq in enumerate(new_sequences):
            if i != n:
                f.write(seq)
                f.write("\n")
            else:
                f.write(seq)
else:
    print("Input and output file must be given")
    print("Input file must be fasta file and can be defined with the argument --input or -i")
    print("Ouput file can be defined with the argument --output or -o")