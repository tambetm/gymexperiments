import argparse
import random
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=100)
parser.add_argument('--options', default='--gym_record')
parser.add_argument('--label_skip', type=int, default=2)
args, remaining_args = parser.parse_known_args()

print args.trials
for n in xrange(args.trials):
  cmd_args = []
  for arg in remaining_args:
    if "," in arg:
      options = arg.split(",")
      choice = random.choice(options)
      cmd_args.append(choice)
    else:
      cmd_args.append(arg)
  label = "_".join(cmd_args[args.label_skip:])
  label = label.replace(' ', '_')
  label = label.replace('--', '_')
  cmd_args.append(args.options)
  cmd_args.append('"'+label+'"')
  cmd_args.append('2>&1')
  cmd_args.append('>'+label+'.log')
  cmd_args = " ".join(cmd_args)
  print cmd_args
  os.system(cmd_args)
  