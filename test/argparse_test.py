import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", default="0", type=int)

args = parser.parse_args()
# print(args.m)
print(args.model)
print(type(args.model))