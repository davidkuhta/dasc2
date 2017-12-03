from __future__ import print_function

import json
import os
import time

from mpyq import MPQArchive

class SC2ReplayFilter(object):
    
    def __init__(self, mmr=1000, apm=10, races=['Zerg','Terr','Prot'], gameMap=None, dataBuild=None):
        self.mmr_threshold = mmr
        self.apm_threshold = apm
        self.races = races
        self.map_title = gameMap
        self.game_version = int(dataBuild) if dataBuild else None

    def _info(self):
        print("\nFiltering for replays with:")
        if self.game_version:
          print("\tGame Build Version:\t"+str(self.game_version))
        if self.map_title:
          print("\tMap:\t\t"+str(self.map_title))
        print("\tMinimum MMR of:\t"+str(self.mmr_threshold))
        print("\tMinimum APM of:\t"+str(self.apm_threshold))
        print("\tRaces:\t\t" + ', '.join(self.races))
        print('')
        
    def filter_replay(self, replay):
        archive =  MPQArchive(replay).extract()
        metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
        
        replay_game_version = int(metadata['DataBuild'])
        if self.game_version:
            if not self.game_version == replay_game_version:
                return None
            
        replay_map_title = metadata['Title']
        if self.map_title:
            if not self.map_title == replay_map_title:
                return None

        players = metadata["Players"]
        for player in players:
            try:
                if player['APM'] < self.apm_threshold or player['MMR'] < self.mmr_threshold or \
                    not player['AssignedRace'] in self.races:
                        return None
            except KeyError:
                return None
        
        return metadata

    def filter_replays(self, directory='./'):
        self._info()

        timestring =    time.strftime("%Y%m%d-%H%M%S")
        filtered_name = "filtered_replays_" + timestring
        json_filename = filtered_name + ".json"
        txt_filename =  filtered_name + ".txt"

        total_replay_counter = 0
        filtered_replay_counter = 0

        with open(json_filename, 'w') as json_file, open(txt_filename, 'w') as txt_file:
            replay_metadata_list = []

            file_list = os.listdir(directory)
            PROGRESS_BAR_FLAG = False

            try:
                from tqdm import tqdm
                file_list = tqdm(file_list)
                PROGRESS_BAR_FLAG = True
            except ImportError:
                pass

            for filename in file_list:

                if filename.endswith(".SC2Replay"):
                    total_replay_counter += 1
                    replay = os.path.join(directory, filename)
                    replay_metadata = self.filter_replay(replay)

                    if replay_metadata:
                        filtered_replay_counter += 1
                        txt_file.write(filename+"\n")
                        replay_json = {str(filename.split(".")[0]):replay_metadata}
                        replay_metadata_list.append(replay_json)

            json.dump(replay_metadata_list, json_file)
                  
        json_file.close()
        txt_file.close()

        if total_replay_counter:
            print(str(filtered_replay_counter) + " of " + str(total_replay_counter) + " replays met filtering criteria\n")
        else:
            print("No SC2 Replays to evaluate in this directory")

        if filtered_replay_counter:
            print("List of replays written to: " + str(txt_filename))
            print("JSON data for replays written to: " + str(json_filename))
        else:
            os.remove(json_filename)
            os.remove(txt_filename)
