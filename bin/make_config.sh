#!/usr/bin/env bash

# ".aisecurity.make_config"
# Program to make config file (~/.aisecurity.json)

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

printf '{\n    "key_directory": "%s/keys/",\n    "key_location": "%s/keys/keys_file.json",\n    "database_location": "%s/database/encrypted.json"\n}\n' \
"$config_path" "$config_path" "$config_path" > "$config_path/aisecurity.json"

if [ ! -d "$config_path/database" ] ; then
  echo "Making database and unknown directories"
  mkdir database
  cd "$config_path/database" || echo "Error: unable to access $config_path/database"
  mkdir unknown
  wget -O "encrypted.json" "https://www.dropbox.com/s/4k11f5c7yhvbc6b/encrypted.json?dl=1" || \
  echo "Error: unable to download encrypted.json"
fi

if [ ! -d "$config_path/models" ] ; then
  echo "Downloading Facenet models"
  cd "$config_path" || echo "Error: unable to access $config_path"
  mkdir models
  cd models || echo "Error: unable to access $config_path/models"
  wget -O "ms_celeb_1m.pb" "https://www.dropbox.com/s/ua0hhioidnhpd90/ms_celeb_1m.pb?dl=1" \
  || echo "Error: MS-Celeb-1M model could not be downloaded"
  wget -O "vgg_face_2.pb" "https://www.dropbox.com/s/rixgtj6pjs9r9ob/vgg_face_2.pb?dl=1" \
  || echo "Error: VGGFace2 model could not be downloaded"
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
  wget -O "drop.sql" "https://github.com/orangese/aisecurity/raw/v1.0a/bin/drop.sql" || \
  echo "Error: drop.sql could not be downloaded"
fi
