#!/usr/bin/env python

import os
import argparse
import subprocess

import sys
sys.path.insert(0, '../')

def extract_replays(archives_dir, replays_dir, password, remove=False):
    # Create directory for replays if it does not exist
    if not os.path.exists(replays_dir):
        os.makedirs(replays_dir)

    if os.path.exists(archives_dir):
        archives = []

        for file in os.listdir(archives_dir):
            if file.endswith(".zip"):
                archives.append(os.path.join(archives_dir, file))

        for archive in archives:
            s = '7z e {0} -o{1} -p{2}'.format(archive, replays_dir, password)
            subprocess.call([s], shell=True)
            if remove:
                os.remove(archive)

import sys

def prompt_for_licensing_agreement(license_agreement):
    """
    This prompt ensures that the user agrees to the Blizzard EULA as denoted
    in s2clientprotocol
    """
    valid = {"iagreetotheeula": "iagreetotheeula", "no": False, "n": False}

    prompt = " [denoted password or 'No'] \n"

    count = 0
    while True:
        sys.stdout.write(license_agreement + prompt)
        choice = raw_input().lower()

        if choice in valid:
            return valid[choice]
        elif count < 5:
            count += 1
            sys.stdout.write("Please respond with the denoted password or 'no'.\n")
        else:
            return False

def parse_args():
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

    password = prompt_for_licensing_agreement(license_agreement)
    if password:
        args = parse_args()
        extract_replays(args.a_dir, args.r_dir, password, args.remove)
    else:
        sys.stdout.write("\n You did not agree to the EULA so no SC2 replays will be extracted from the archives\n")
        return

if __name__ == '__main__':
    main()
