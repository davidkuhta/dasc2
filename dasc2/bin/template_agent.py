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

"""Copies SC2 Agent templates to the designed directory"""

import os
import shutil
import argparse
import time

from pkg_resources import resource_filename

def template_agent(agent, agents_dir='./agents', name=None):
    """Function which duplicates an agent template
        Args:
            agent (str):            Agent to be templated
            agents_dir (str):       Directory in which to place agents
            name (str, optional):   Name to rename templated agent         
            remove (bool):      Remove the archive files upon extraction.
    """
    agents = {
                "dasc2" : resource_filename(__name__, '../agent/dasc2_agent.py'),
                "base": resource_filename('pysc2', 'agents/base_agent.py'),
                "random": resource_filename('pysc2', 'agents/random_agent.py'),
                "scripted": resource_filename('pysc2', 'agents/scripted_agent.py')
             }
    if not os.path.exists(agents_dir):
        os.makedirs(agents_dir)

    agents_path = os.path.abspath(agents_dir)
    outputs_dir = os.path.join(agents_path, 'outputs/')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    if agent in agents:
        agent_file = agents[agent]
        old_agent_basename = os.path.basename(agent_file)
        old_agent_file = os.path.join(agents_dir, old_agent_basename)

        if os.path.isfile(old_agent_file):
            tmp_dir = os.path.join(agents_path, 'tmp/')
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            tmp_file = os.path.join(tmp_dir, old_agent_basename)    
            shutil.copyfile(agent_file, tmp_file)

            timestring = time.strftime("%m%d%Y%H%M%S")
            renamed_basename = timestring + "_" + old_agent_basename
            old_agent_file = os.path.join(agents_dir, renamed_basename)
            os.rename(tmp_file, old_agent_file)
            os.rmdir(tmp_dir)
        else:
            shutil.copyfile(agent_file, old_agent_file)
        
        if name:
            new_agent_filename = name + ".py"
            new_agent_file = os.path.join(agents_dir, new_agent_filename)

            os.rename(old_agent_file, new_agent_file)
    else:
        print("That agent is not available to template")


def parse_args():
    """Helper function to parse dasc2_template_agent arguments"""
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', dest='agent', action='store', type=str, default='dasc2',
                        help='The desired agent template to copy', required=True)
    parser.add_argument('--agents_dir', dest='a_dir', action='store', type=str, default='./agents',
                        help='Directory in which to generate agent templates')
    parser.add_argument('--name', dest='name', action='store', type=str,
                        help='Desired file name for new template')
    
    return parser.parse_args()

def main():
    args = parse_args()

    template_agent(args.agent, args.a_dir, args.name)

if __name__ == '__main__':
    main()