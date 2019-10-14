#!/usr/bin/env bash

# ".aisecurity.create_config"
# Program to create config file (~/.aisecurity.json)

if [ ! -d "$HOME/.aisecurity" ] ; then
  mkdir "$HOME/.aisecurity"
else
  read -rp "$HOME/.aisecurity already exists. Overwrite? (y/n): " confirm
  if ! [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] ; then
    echo "Exiting..." ; exit 1
  fi
fi

cd "$HOME/.aisecurity" || echo "Error: unable to access ~/.aisecurity"
config_path=$(realpath .)

echo "Adding aisecurity.json to .aisecurity"
touch "$HOME/.aisecurity/aisecurity.json"

printf '{\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/encrypted.json"\n}\n' \
"$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"

if [ ! -d "$config_path/database" ] ; then
  mkdir database
  cd "$HOME/.aisecurity/database" || echo "Error: unable to access ~/.aisecurity/database"
  mkdir unknown
fi

if [ ! -d "$HOME/.aisecurity/keys" ] ; then
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir keys
fi
