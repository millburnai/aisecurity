#!/usr/bin/env bash

# ".aisecurity.make_config"
# Program to make config files (~/.aisecurity)

if [ ! -d "$HOME/.aisecurity" ] ; then
  mkdir "$HOME/.aisecurity"
else
  read -rp "$HOME/.aisecurity already exists. Overwrite? (y/n): " confirm
  if ! [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] ; then
    echo "Exiting..." ; exit 1
  fi
fi

cd "$HOME/.aisecurity" || echo "Error: unable to access ~/.aisecurity"
config_path=$(pwd )

echo "Adding aisecurity.json to .aisecurity"
touch "$HOME/.aisecurity/aisecurity.json"

printf '{\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/test.json",\n    "mysql_user": "root",\n    "mysql_password": "root"\n}\n' \
"$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"

if [ ! -d "$config_path/database" ] ; then
  echo "Making database and unknown directories"
  mkdir database
  cd "$config_path/database" || echo "Error: unable to access $config_path/database"
  wget -O "test.json" https://www.dropbox.com/s/uhii2vj373y7odj/test.json?dl=0 || echo "Error: unable to download test.json"
fi

if [ ! -d "$config_path/models" ] ; then
  echo "Downloading Facenet models"
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir models
  cd models || echo "Error: unable to access $config_path/models"
  wget -O "ms_celeb_1m.h5" "https://www.dropbox.com/s/i4r3jbnzuzcc9fh/ms_celeb_1m.h5?dl=1" \
  || echo "Error: MS-Celeb-1M model could not be downloaded"
  wget -O "vgg_face_2.h5" "https://www.dropbox.com/s/4xo8uuhu9ug8ir3/vgg_face_2.h5?dl=1" \
  || echo "Error: VGGFace2 model could not be downloaded"
  wget -O "haarcascade_frontalface_default.xml" "https://www.dropbox.com/s/zhb4cn9idl6rrvm/haarcascade_frontalface_default.xml?dl=1" \
  || echo "Error: haarcascade model could not be downloaded"
fi

if [ ! -d "$HOME/.aisecurity/keys" ] ; then
  echo "Creating keys directory"
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir keys
fi

if [ ! -d "$HOME/.aisecurity/bin" ] ; then
  echo "Creating bin directory"
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir bin
  cd "$config_path/bin" || echo "Error: unable to access $config_path/bin"
  wget -O "drop.sql" "https://github.com/orangese/aisecurity/raw/v0.9a/bin/drop.sql" || \
  echo "Error: drop.sql could not be downloaded"
  wget -O "dump_embeds.sh" "https://raw.githubusercontent.com/orangese/aisecurity/v0.9a/bin/dump_embeds.sh" || \
  echo "Error: dump_embeds.sh could not be downloaded"
fi

if [ ! -d "$HOME/.aisecurity/logging" ] ; then
  echo "Creating logging directory"
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir database
  cd "$config_path/logging" || echo "Error: unable to access $config_path/logging"
  touch firebase.json
  echo "Fill in '$config_path/logging/firebase.json' and a key file in the same directory to use firebase logging"
  mkdir unknown
fi
