# daSC2 - Data Analytics for StarCraft II

[daSC2](https://github.com/davidkuhta/dasc2) is a data analytics toolset
under development for StarCraft II.

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

## Filter replays

This tool filters `.SC2Replay` files within the current working directory,
generating two files containing information related to the replay files meeting
the filtering criteria:
  * `filtered_replays_timestamp.txt` - list of replays files meeting filtering criteria.
  * `filtered_replays_timestamp.json` - list of dictionaries of the replay metadata (replay file name as the key).

You can filter a directory of replays using the command:

```shell
$ python -m dasc2.bin.filter_replays
```

Note: `dasc2_filter` can be used as a shortcut to `python -m dasc2.bin.replay_filter`.

This filters `.SC2Replay` files within the current working directory.

### Optional Flags
|flag          |description							|default					|
|--------------|------------------------------------|---------------------------|
|`-h, --help`  |Show help message					|							|
|`--replay_dir`|Select replay directory				|current working directory	|
|`--min_mmr`   |Set the minimum player MMR 			|1000						|
|`--min_apm`   |Set the minimum player APM 			|10							|
|`--races`     |Filter on certain races 			|Zerg Terr Prot 			|
|`--game_map`  |Filter based on map title|all maps	|							|
|`--build`     |Filter based on game build version	|							|

Note: no commas needed between race names
# Replays

## Obtaining Replays

1. You can evaluate your own replays
2. Blizzard
	1. Download replay packs linked on the [s2client-proto](https://github.com/Blizzard/s2client-proto#downloads) repo
	2. Download via the [replay-api](https://github.com/Blizzard/s2client-proto/tree/master/samples/replay-api)