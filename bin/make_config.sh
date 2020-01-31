#!/usr/bin/env bash

# ".aisecurity.make_config"
# Program to make config files (~/.aisecurity)

if [ ! -d "$HOME/.aisecurity" ] ; then
  mkdir "$HOME/.aisecurity"
fi

cd "$HOME/.aisecurity" || echo -e "\e[31mError: unable to access ~/.aisecurity\e[0m"
config_path=$(pwd )

if [ ! -f "$HOME/.aisecurity/aisecurity.json" ] ; then
  echo -e "\e[95mAdding aisecurity.json to .aisecurity\e[0m"
  touch "$HOME/.aisecurity/aisecurity.json"
  printf '{\n    "default_model": "%s/models/ms_celeb_1m.h5",\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/test.json",\n    "database_info": "%s/database/test_info.json",\n    "mysql_user": "root",\n    "mysql_password": "root"\n}\n' \
  "$config_path" "$config_path" "$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"
fi

if [ ! -d "$config_path/database" ] ; then
  echo -e "\e[95mMaking database and unknown directories\e[0m"
  mkdir database
  cd "$config_path/database" || echo -e "\e[31mError: unable to access $config_path/database\e[0m"
  curl -Lo "test.json" "https://www.dropbox.com/s/umjku76xppc0396/test.json?dl=1" \
  || echo -e "\e[31mError: unable to download test.json\e[0m"
  curl -Lo "test_info.json" "https://www.dropbox.com/s/ihfmemt6sqdfj74/test_info.json?dl=1" \
  || echo -e "\e[31mError: unable to download test_info.json\e[0m"
fi

if [ ! -d "$config_path/models" ] ; then
  echo -e "\e[95mDownloading Facenet models\e[0m"
  cd "$config_path" || echo -e "\e[31mError: unable to access $config_path\e[0m"
  mkdir models
  cd models || echo -e "\e[31mError: unable to access $config_path/models\e[0m"
  curl -Lo "ms_celeb_1m.h5" "https://www.dropbox.com/s/i4r3jbnzuzcc9fh/ms_celeb_1m.h52?dl=1" \
  || echo -e "\e[31mError: MS-Celeb-1M model could not be downloaded\e[0m"
  curl -Lo "haarcascade_frontalface_default.xml" "https://www.dropbox.com/s/zhb4cn9idl6rrvm/haarcascade_frontalface_default.xml?dl=1" \
  || echo -e "\e[31mError: haarcascade model could not be downloaded\e[0m"
fi

if [ ! -d "$HOME/.aisecurity/keys" ] ; then
  echo -e "\e[95mCreating keys directory\e[0m"
  cd "$config_path" || echo -e "\e[31mError: unable to access $config_path\e[0m"
  mkdir keys
fi

if [ ! -d "$HOME/.aisecurity/bin" ] ; then
  echo -e "\e[95mCreating bin directory\e[0m"
  cd "$config_path" || echo -e "\e[31mError: unable to access $config_path\e[0m"
  mkdir bin
  cd "$config_path/bin" || echo -e "\e[31mError: unable to access $config_path/bin\e[0m"
fi

if [ ! -d "$HOME/.aisecurity/logging" ] ; then
  echo -e "\e[95mCreating logging directory\e[0m"
  cd "$config_path" || echo -e "\e[31mError: unable to access $config_path\e[0m"
  mkdir logging
  cd "$config_path/logging" || echo -e "\e[31mError: unable to access $config_path/logging\e[0m"
  touch "$config_path/logging/firebase.json"
  echo -e "\e[31mFill in '$config_path/logging/firebase.json' and a key file in the same directory to use firebase logging\e[0m"
  mkdir unknown
fi

if [ ! -d "$HOME/.aisecurity/config" ] ; then
  echo -e "\e[95mCreating config directory\e[0m"
  cd "$config_path" || echo -e "\e[31mError: unable to access $config_path\e[0m"
  mkdir config
  cd "$config_path/config" || echo -e "\e[31mError: unable to access $config_path/config\e[0m"
  curl -Lo "models.json" "https://www.dropbox.com/s/9my8ofbzohi0dsm/models.json?dl=1" \
  || echo -e "\e[31mError: unable to download models.json\e[0m"
  curl -Lo "cuda_models.json" "https://www.dropbox.com/s/ieke59ny0r7qxo3/cuda_models.json?dl=1" \
  || echo -e "\e[31mError: unable to download cuda_models.json\e[0m"
fi

echo -e "\e[96m~/.aisecurity is up to date\e[0m"
