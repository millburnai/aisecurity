#!/usr/bin/env bash

# ".aisecurity.update_config"
# Program to update config file (~/PycharmProjects/aisecurity/.aisecurity/aisecurity.json) based on local paths

if [ -d "$HOME/PycharmProjects/aisecurity/" ] ; then
  config_path="$HOME/PycharmProjects/aisecurity/.aisecurity"
  printf '{\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/encrypted.json"\n}\n' \
"$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"
elif [ -d "$HOME/Desktop/aisecurity/" ] ; then
  config_path="$HOME/Desktop/aisecurity/.aisecurity"
  printf '{\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/encrypted.json"\n}\n' \
"$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"
else
  echo "Error: aisecurity repository not found" ; exit 1
fi
