#!/usr/bin/env python
#
# Portions of code reused from following link in accordance with license below:
# https://github.com/Blizzard/s2client-proto/blob/master/samples/replay-api/download_replays.py
#
# Copyright (c) 2017 Blizzard Entertainment
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Downloads SC2Replay files using the Blizzard API"""

import os
import requests
import json
import argparse
import shutil

import six
from six.moves.urllib import parse as urlparser

from dasc2.lib.replay_helpers import check_build_version

API_BASE_URL = 'https://us.api.battle.net'
API_NAMESPACE = 's2-client-replays'


def get_bnet_oauth_access_token(url, key, secret):
    headers = { "Content-Type": "application/json"}
    params = {
        "grant_type": "client_credentials",
        "client_id" : key,
        "client_secret" : secret
    }
    response = requests.post(url=url, headers=headers, params=params)
    response = json.loads(response.text)
    if 'access_token' in response:
        return response['access_token']
    raise Exception('Failed to get oauth access token. response={}'.format(response))


def get_base_url(access_token):
    headers = {"Authorization": "Bearer " + access_token}
    params = {
        'namespace' : API_NAMESPACE,
    }
    response = requests.get(urlparser.urljoin(API_BASE_URL, "/data/sc2/archive_url/base_url"), headers=headers,
                          params=params)
    return json.loads(response.text)["base_url"]


def search_by_client_version(access_token, client_version):
    headers = {"Authorization": "Bearer " + access_token}
    params = {
        'namespace' : API_NAMESPACE,
        'client_version' : client_version,
        '_pageSize' : 25
    }
    response = requests.get(urlparser.urljoin(API_BASE_URL, "/data/sc2/search/archive"), headers=headers, params=params)
    response = json.loads(response.text)
    meta_urls = []
    for result in response['results']:
        assert result['data']['client_version'] == client_version
        meta_urls.append(result['key']['href'])
    return meta_urls


def get_meta_file_info(access_token, url):
    headers = { "Authorization": "Bearer " + access_token}
    params = {
        'namespace' : API_NAMESPACE,
    }
    response = requests.get(url, headers=headers, params=params)
    return json.loads(response.text)


def download_file(url, output_dir):
    # Create output_dir if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = url.split('/')[-1]
    file_path = os.path.join(output_dir, file_name)

    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    return file_path


def get_replay_pack(client_version, client_key, client_secret, output_dir, extract=False):
    try:
        # Get OAuth token from us region
        access_token = get_bnet_oauth_access_token("https://us.battle.net/oauth/token", client_key, client_secret)

        # Get the base url for downloading replay packs
        download_base_url = get_base_url(access_token)

        # Get meta file infos for the give client version
        print('Searching replay packs with client version=' + client_version)
        meta_file_urls = search_by_client_version(access_token, client_version)
        if len(meta_file_urls) == 0:
            print('No matching replay packs found for the client version!')
            return

        # For each meta file, construct full url to download replay packs
        print('Building urls for downloading replay packs. num_files={0}'.format(len(meta_file_urls)))
        download_urls=[]
        for meta_file_url in meta_file_urls:
            meta_file_info = get_meta_file_info(access_token, meta_file_url)
            file_path = meta_file_info['path']
            download_urls.append(urlparser.urljoin(download_base_url, file_path))

        
        files = []

        sorted_urls = sorted(download_urls)

        # We'll try to use tqdm for pretty printing a status bar
        # which will succeed if the user selected to install it
        # during the `pip install dasc2`
        try:
            from tqdm import tqdm
            sorted_urls = tqdm(sorted_urls)
        except ImportError:
            pass

        # Download replay packs.
        for archive_url in sorted_urls:
            print('Downloading replay packs. url='  + archive_url)
            files.append(download_file(archive_url, output_dir))

    except Exception as e:
        import traceback
        print('Failed to download replay packs. traceback={}'.format(traceback.format_exc()))


def parse_args():
    """Helper function to parse dasc2_download arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--key', dest='client_key', action='store', type=str,
                        help='Battle.net API key', required=True)
    parser.add_argument('--secret', dest='client_secret', action='store', type=str,
                        help='Battle.net API secret', required=True)
    parser.add_argument('--version', dest='s2_client_version', action='store', type=str,
                        help='Starcraft2 client version for searching replay archives with', required=True)
    parser.add_argument('--archive_dir', dest='a_dir', action='store', type=str, default='./archives',
                        help='the directory where the downloaded replay archives will be saved to')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Process user inputed build_version to allow for version (ex: 58977) or label (ex: "4.0.2")
    processed_version = str(check_build_version(args.s2_client_version, True))

    get_replay_pack(processed_version, args.client_key, args.client_secret, args.a_dir)


if __name__ == '__main__':
    main()