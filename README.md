# daSC2 - Data Analytics for StarCraft II

[daSC2](https://github.com/davidkuhta/dasc2) is a data analytics toolset
under development for StarCraft 2 by David Kuhta and Anshul Sacheti.

# Quick Start Guide

## Get daSC2

### PyPI

The easiest way to get daSC2 is to use pip:

```shell
$ pip install dasc2
```

or if you desire a visual progress bar:

```shell
$ pip install dasc2[with-progress-bar]
```

That will install the `dasc2` package along with all the required dependencies.

Pip will install a few of the binaries to your bin directory.

### Git

Alternatively you can install daSC2 with git.

1.  Clone the daSC2 repo
2.  Install the dependencies and `dasc2` package:

```shell
$ git clone https://github.com/davidkuhta/dasc2.git
$ pip install dasc2/
```

# Available Commands

## Download replays
This tool allows for the downloading of .SC2Replay archives via Blizzard's API.

You can download replays using the command:
`dasc2_download`

### Required Flags
|flag          |description							        |
|--------------|--------------------------------|						
|`--key`       |Blizzard API client key		      |
|`--secret`    |Blizzard API client secret 			|
|`--version`   |Desire StarCraft 2 Build Version|

### Optional Flags
|flag            |description							           | default     |
|----------------|-----------------------------------|-------------|
|`-h, --help`    |Show help message	 				         |             |
|`--archives_dir`|Directory in which to save archives|`./archives`|

## Extract Replays
This tool simplifies extracting of the StarCraft replay files from the previously
download archives.

You can extract replays using the command:
`dasc2_extract`

In accordance with Blizzard's EULA you will be prompted for agreement to the
[Blizzard AI and Machine Learning License](http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html)

### Optional Flags
|flag               |description							                              | default     |
|-------------------|-------------------------------------------------------|-------------|
|`-h, --help`       |Show help message	 				                            |             |
|`--archives_dir`   |Directory in which archives are located                |`./archives` |
|`--replays_dir`    |Directory in which to extract replays                  |`./replays`  |
|`--remove_archives`|Flag denoted to remove the archives upon extraction    |             |

## Filter replays

This tool filters `.SC2Replay` files within the denoted directory,
generating a filter file containing information related to the replay files meeting
the filtering criteria:

You can filter replays using the command:
`dasc2_filter`

### Optional Flags
|flag          |description							|default					|
|--------------|------------------------------------|---------------------------|
|`-h, --help`  |Show help message					|							|
|`--replays_dir`|Select replay directory				|`./replays`|
|`--filters_dir`|Select directory to place filters				|`./filters`|
|`--min_mmr`   |Set the minimum player MMR 			|1000						|
|`--min_apm`   |Set the minimum player APM 			|10							|
|`--races`     |Filter on certain races 			|Zerg Terr Prot 			|
|`--winning_races`     |Filter on whether the winner belongs to a certain race 			|Zerg Terr Prot 			|
|`--game_map`  |Filter based on map title|all maps	|							|
|`--build`     |Filter based on game build version	|							|
|`--full_path` |Flag denoting filters should list full path of replays	|							|

Note: no commas needed between race names

## Generate replay states

This tool generates replay states for a list of replays denoted in a filter file.
Note corrupted replays will be skipped so yield may be less than 100% as compared
to the input list.

You can generate replay states using the command:
`dasc2_states`

### Required Flags
|flag          |description							                   |
|--------------|-------------------------------------------|						
|`--replay_list`|File containing list of replays to evaluate|
|`--filter_file`|File cotaining filter information				|

### Optional Flags
|flag          |description							|default					|
|--------------|------------------------------------|---------------------------|
|`-h, --help`  |Show help message					|							|
|`--parallel`|How many instances to run in parallel				|1|
|`--step_mul`|How many game steps per observation|1|
|`--states_dir`   |Directory to save states|./states						|
|`--print_time`   |Interval between stat prints and data saves in seconds|10							|
|`--winner_only`  |Generate states for winner only 			|False	|

## Build Orders

This tool generates json files containing replay information including build orders
for both a player and their opponent

You can generate replay states using the command:
`dasc2_build_orders`

### Required Flags
|flag          |description							                   |
|--------------|-------------------------------------------|						
|`--replay_list`|File containing list of replays to evaluate|

### Optional Flags
|flag          |description							|default					|
|--------------|------------------------------------|---------------------------|
|`-h, --help`  |Show help message					|							|
|`--states_dir`|Directory where states are located				|`./states`|
|`--build_orders_dir`|Directory in which to generate build orders|`./build_orders`|

## Template Agent

This tool generates a duplicate of existing agents for use as templates.

Available agents include:

* dasc2 (An RL agent)
* base (pysc2's base agent)
* scripted (pysc2's scripted agent)
* random (pysc2's random agent)

You can generate replay states using the command:
`dasc2_template_agents`

### Required Flags
|flag          |description							                   |
|--------------|-------------------------------------------|						
|`--agent`|Agent to create a template of.|

### Optional Flags
|flag          |description							|default					|
|--------------|------------------------------------|---------------------------|
|`-h, --help`  |Show help message					|							|
|`--agents_dir`|Directory where agents are to be generated				|`./agents`|
|`--name`|The filename to rename the agent to|`./build_orders`|

# Replays

## Obtaining Replays

1. `dasc2_download`
2. You can evaluate your own replays
3. Blizzard
	1. Download replay packs linked on the [s2client-proto](https://github.com/Blizzard/s2client-proto#downloads) repo

# Attribution

We'd like to thank [Blizzard](https://github.com/Blizzard/s2client-proto/),
[Deepmind](https://github.com/deepmind/pysc2), [Arthur Juliani](https://github.com/awjuliani), and [Cole Maclean](https://github.com/cole-maclean/autocraft),
for the influence each of their projects had on daSC2. Any code utilized
was in accordance with it's respective license.
