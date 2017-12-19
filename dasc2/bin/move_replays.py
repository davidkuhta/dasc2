#!/usr/bin/env python
#
# Copyright 2017 David Kuhta & Anshul Sacheti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

from shutil import copyfile

def parse_args():
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
