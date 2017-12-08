#!/usr/bin/env python

import os
from shutil import copyfile

from dasc2.lib.replay_filter import SC2ReplayFilter


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--replayFileList', dest='rFile', action='store', default=None, help='File containing list of replays',required=True)
    parser.add_argument('--outputDir', dest='outputDir', action='store', default=None, help='Directory to copy replays to',required=True)
    
    return parser.parse_args()

def main():
  args = parse_args()

  # if directory doesn't exist, attempt to create it
  if not(os.path.isdir(args.outputDir)):
    os.makedirs(args.outputDir)
    
  # try looking again but exit if path bad (can't copy files)
  if not(os.path.isdir(args.outputDir)):
  	raise ValueError("Output directory doesn't exist")

  PROGRESS_BAR_FLAG = False
  f = open(args.rFile, 'r')
  replayList = list(f)
  f.close()

  fileCount = len(replayList)

  try:
    from tqdm import tqdm
    replayList = tqdm(replayList)
    PROGRESS_BAR_FLAG = True
  except ImportError:
    pass
  
  for replayPath in replayList:
    replayPath = replayPath.rstrip()
    copyfile(replayPath,os.path.join(args.outputDir, replayPath.split("/")[-1]))

  print("%i Files copied to %s" % (fileCount, args.outputDir))

if __name__ == '__main__':
    main()
