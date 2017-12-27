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

"""Extracts SC2Replay files from .zip archives"""

import os
import argparse
import subprocess

import sys
sys.path.insert(0, '../')

import six
from six.moves import input

def extract_replays(archives_dir, replays_dir, password, remove=False):
    """Function which extracts replays
        Args:
            archives_dir (str): The directory which contains the archives to be extracted.
            replays_dir (str):  The directory in which to extract the replay files.
            password (str):     Agreement or Disagreement with EULA.
            remove (bool):      Remove the archive files upon extraction.
    """

    if os.path.exists(archives_dir):
        # Create user specified replay directory
        if not os.path.exists(replays_dir):
            os.makedirs(replays_dir)
        
        # Initialize empty list to hold archives
        archives = []

        # Iterate through archive files
        # If file is a zip file, append to archives list
        for file in os.listdir(archives_dir):
            if file.endswith(".zip"):
                archives.append(os.path.join(archives_dir, file))

        # Iterate through generated archive list
        # calling the 7z util to extract the file
        for archive in archives:
            s = '7z e {0} -o{1} -p{2}'.format(archive, replays_dir, password)
            subprocess.call([s], shell=True)
            if remove:
                os.remove(archive)

def __prompt_for_licensing_agreement(license_agreement):
    """Confirm user agreement to Blizzard EULA
    This function provides a prompt for the user to agree to, 
    in respect for the Blizzard EULA as denoted in s2clientprotocol
    
    Args:
        license_agreement (str):  license_agreement to presented to user.
    """

    # Instantiate valid options dict
    valid_option = {"iagreetotheeula": "iagreetotheeula", "no": False, "n": False}

    # Create user prompt
    prompt = " [denoted password or 'No'] \n\n"

    # Intialize counter
    count = 0

    while True:
    # Output the license_agreement and prompt
        sys.stdout.write(license_agreement + prompt)
    # Assign user input to choice
        #REMOVING
        # try:
        #     choice = raw_input().lower()
        # except NameError:
        #     choice = input().lower()
        choice = input().lower()

    # If user entires a valid option return it's corresponding value
        if choice in valid_option:
            return valid_option[choice]
    # Else provide the user with three additional attempts
        elif count < 3:
            count += 1
            sys.stdout.write("Please respond with the denoted password or 'no'.\n\n")
        else:
            return False

def parse_args():
    """Helper function to parse dasc2_extract arguments"""
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--archives_dir', dest='a_dir', action='store', type=str, default='./archives',
                        help='Directory containing replay archives')
    parser.add_argument('--replays_dir', dest='r_dir', action='store', type=str, default='./replays',
                        help='Directory in which to extract replays')
    parser.add_argument('--remove_archives', dest='remove', action='store_true',
                        help='remove archives after extraction')
    
    return parser.parse_args()

def main():
    license_agreement = """
    You may find zipped files under the replays directory if there's any
    matching replay packs for the given S2 client version. To access Starcraft 2
    replay packs, you must agree to the AI and Machine Learning License The
    files are password protected with the password 'iagreetotheeula'.
    By typing in the password 'iagreetotheeula' you agree to be bound by the
    terms of the AI and Machine Learning License which can be found at
    http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html \n """

    password = __prompt_for_licensing_agreement(license_agreement)
    if password:
        args = parse_args()
        extract_replays(args.a_dir, args.r_dir, password, args.remove)
    else:
        sys.stdout.write("\n You did not agree to the EULA so no SC2 replays will be extracted from the archives\n")
        return

if __name__ == '__main__':
    main()