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

"""A module containing helper functions for manipulating replays"""

import json

from pkg_resources import resource_filename

def __get_label(versions, build_label, is_label):
	"""Evaluates the provided build_label and returns if supported

	    Args:
	        version:		A dictionary containing a list of supported build version.
	        build_label:	The user entered build_label.
	        is_label:		Boolean: Whether the provided build_label is a label or a version.
	    
	    Returns:
	        int or string depending on return_label

	    Raises error in the event of an unsupported build type
	"""
	# Check if supplied build matches a build version label
	if is_label:
		if any(v['label'] == str(build_label) for v in versions):
			return build_label
		raise ValueError('Build label not supported')
	else: 
	    # Check if supplied build version matches a 'version'
		build_label = next((v['label'] for v in versions if (v['version'] == int(build_label))), None)
		if build_label:
			return build_label
		raise ValueError('Build version not supported')

def __get_version(versions, build_version, is_version):
	"""Evaluates entered build_version and returns if supported

	    Args:
	        version:		A dictionary containing a list of supported build version.
	        build_version:	The user entered build_version.
	        is_version:		Boolean: Whether the provided build_version is a version or a label.
	    
	    Returns:
	        int or string depending on return_label

	    Raises error in the event of an unsupported build type
	"""
	if is_version:
	    # Check if supplied build matches a version
		if any(v['version'] == int(build_version) for v in versions):
			return build_version
		raise ValueError('Build version not supported')
	else: 
	    # Check if supplied build version matches a label
		build_version = next((v['version'] for v in versions if (v['label'] == str(build_version))), None)
		if build_version:
			return build_version
		raise ValueError('Build label not supported')

def check_build_version(build_version, return_label):
	"""Evaluates the provided build_version against supported versions

	    Args:
	        build_version:	A string corresponding to a build version.
	        				or build label. Ex: "59877" or "4.0.2"
	        return_label:	Boolean, whether to return a label (TRUE) or version (FALSE).
	    
	    Returns:
	        int or string depending on return_label

	    Raises error in the event of an unsupported build type
	"""
    # Process the build_versions JSON file
	version_file = open(resource_filename(__name__, '../data/build_versions.json')).read()
	versions = json.loads(version_file)

	# Cast build_version to int if it's an int
	try:
		build_version = int(build_version)
	except ValueError:
		pass

	# Process the build_version based on whether it's a str/int and the desired return value
	if type(build_version) is str and return_label:
		# build_version is a label, and we want a label returned
		return __get_label(versions, build_version, True)
		# build_version is a label, but we want a version returned
	elif type(build_version) is str:
		return __get_version(versions, build_version, False)
	elif type(build_version) is int and return_label:
		# build_version is a version, but we want a label returned
		return __get_label(versions, build_version, False)
	elif type(build_version) is int:
		# build_version is a version, and we want a version returned
		return __get_version(versions, build_version, True)
	else:
			# build version isn't a string or integer 
		raise TypeError('Incorrect build version type')