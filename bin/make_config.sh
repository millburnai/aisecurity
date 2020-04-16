#!/bin/bash

# ".aisecurity.make_config"
# Program to make config files (~/.aisecurity)

ERRORS=0

# get home path
if [[ $OSTYPE == "darwin"* ]] || [ "$OSTYPE" = "linux-gnu" ] ; then
  ROOT=$HOME
elif [[ $OSTYPE == "msys" ]] ; then
  ROOT=$HOMEPATH;
else
  echo "$OSTYPE not supported" && exit 1;
fi

# .aisecurity
if [ ! -d "$ROOT/.aisecurity" ] ; then
  echo -e "\033[0;95mCreating ~/.aisecurity\033[0m"
  mkdir "$ROOT/.aisecurity"
fi

cd "$ROOT/.aisecurity/" || { echo -e "\033[0;31mError: unable to access ~/.aisecurity\033[0m" ; exit 1; }
config_path=$(pwd)
if [[ $OSTYPE == "msys" ]] ; then
  config_path="${config_path/\/c\//C:/}"
fi

# aisecurity.json
if [ ! -f "$config_path/aisecurity.json" ] ; then
  echo -e "\033[0;95mCreating ~/.aisecurity/aisecurity.json\033[0m"
  touch "$config_path/aisecurity.json"
  printf '{\n    "default_model": "%s/models/ms_celeb_1m.h5",\n    "name_keys": "%s/keys/test_name_keys.txt",\n    "embedding_keys": "%s/keys/test_embedding_keys.txt",\n    "database_location": "%s/database/test.json",\n    "database_info": "%s/database/test_info.json",\n    "mysql_user": "root",\n    "mysql_password": "root"\n}\n' \
  "$config_path" "$config_path" "$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"
fi

# database
if [ ! -d "$config_path/database" ] ; then
  echo -e "\033[0;95mCreating database and unknown directories\033[0m"
  mkdir "$config_path"/database
fi

if [ ! -f "$config_path/database/test.json" ] ; then
  echo -e "\033[0;95mDownloading mini-database\033[0m"
  wget -O "$config_path/database/test.json" "http://www.dropbox.com/s/umjku76xppc0396/test.json?dl=1" \
  || { echo -e "\033[0;31mError: unable to download test.json\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

if [ ! -f "$config_path/database/test_info.json" ] ; then
  echo -e "\033[0;95mDownloading mini-database config\033[0m"
  wget -O "$config_path/database/test_info.json" "https://www.dropbox.com/s/ihfmemt6sqdfj74/test_info.json?dl=1" \
  || { echo -e "\033[0;31mError: unable to download test_info.json\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi


# models
if [ ! -d "$config_path/models" ] ; then
  echo -e "\033[0;95mCreating model directory\033[0m"
  mkdir models
fi

if [ ! -f "$config_path/models/ms_celeb_1m.h5" ] ; then
  echo -e "\033[0;95mDownloading MS-CELEB-1M keras model\033[0m"
  wget -O "$config_path/models/ms_celeb_1m.h5" "https://www.dropbox.com/s/i4r3jbnzuzcc9fh/ms_celeb_1m.h52?dl=1" \
  || { echo -e "\033[0;31mError: MS-Celeb-1M model could not be downloaded\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

if [ ! -f "$config_path/models/haarcascade_frontalface_default.xml" ] ; then
  echo -e "\033[0;95mDownloading Haarcascade model\033[0m"
  wget -O "$config_path/models/haarcascade_frontalface_default.xml" \
  "https://www.dropbox.com/s/zhb4cn9idl6rrvm/haarcascade_frontalface_default.xml?dl=1" \
  || { echo -e "\033[0;31mError: haarcascade model could not be downloaded\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

if [ ! -f "$config_path/models/20180402-114759.pb" ] ; then
  echo -e "\033[0;95mDownloading 20180402-114759 tensorflow model\033[0m"
  wget -O "$config_path/models/20180402-114759.pb" "https://www.dropbox.com/s/ek2y33ntzfr2zgq/20180402-114759.pb?dl=1" \
  || { echo -e "\033[0;31mError: 20180402-114759 model could not be downloaded\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

# keys
if [ ! -d "$config_path/keys" ] ; then
  echo -e "\033[0;95mCreating keys directory\033[0m"
  mkdir "$config_path"/keys
fi

if [ ! -f "$config_path/keys/test_name_keys.txt" ] ; then
  echo -e "\033[0;95mDownloading test_name_keys.txt\033[0m"
  wget -O "$config_path/keys/test_name_keys.txt" \
  "https://www.dropbox.com/s/yxebmo4gm0qq7nj/test_name_keys.txt?dl=1" || \
  { echo -e "\033[0;31mError: could not download test_name_keys.txt\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

if [ ! -f "$config_path/keys/test_embedding_keys.txt" ] ; then
  echo -e "\033[0;95mDownloading test_embedding_keys.txt\033[0m"
  wget -O "$config_path/keys/test_embedding_keys.txt" \
  "https://www.dropbox.com/s/kl5s77evy8m9mpm/test_embedding_keys.txt?dl=1" \
  || { echo -e "\033[0;31mError: could not download test_embedding_keys.txt\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

# logging
if [ ! -d "$config_path/logging" ] ; then
  echo -e "\033[0;95mCreating logging directory\033[0m"
  mkdir "$config_path"/logging
  touch "$config_path/logging/firebase.json"
  echo -e "\033[0;31mFill in '$config_path/logging/firebase.json' and a key file in the same directory to use firebase logging\033[0m"
  mkdir "$config_path"/logging/unknown
fi

# config
if [ ! -d "$config_path/config" ] ; then
  echo -e "\033[0;95mCreating config directory\033[0m"
  mkdir "$config_path"/config
fi

if [ ! -f "$config_path/config/models.json" ] ; then
  wget -O "$config_path/config/models.json" "https://www.dropbox.com/s/9my8ofbzohi0dsm/models.json?dl=1" \
  || { echo -e "\033[0;31mError: unable to download models.json\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

if [ ! -f "$config_path/config/cuda_models.json" ] ; then
  wget -O "$config_path/config/cuda_models.json" "https://www.dropbox.com/s/ieke59ny0r7qxo3/cuda_models.json?dl=1" \
  || { echo -e "\033[0;31mError: unable to download cuda_models.json\033[0m" ; ERRORS=$((ERRORS + 1)) ; }
fi

echo -e "\033[0;96m~/.aisecurity update finished with $ERRORS error(s)\033[0m"
