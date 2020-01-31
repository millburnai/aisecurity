#!/usr/bin/env bash

# "aisecurity.security.make_keys"
# Program to create key directory and files
# args: key_dir (directory at which to create keys-- $HOME implied)
#       path_to_json (path to json key list file-- $HOME implied)

key_dir="$1"
path_to_json="$2"

if [ ! -d "$key_dir" ] ; then
  # default locations
  key_dir="$HOME/.aisecurity/keys"
  path_to_json="$HOME/.aisecurity/keys/keys_file.json"
fi

function mk_key_files {
  # Makes key files
  # args: key_loc (key directory)

  key_loc=$1

  if [ ! -f "$key_loc/test_name_keys.txt" ] ; then
    curl -Lo "test_name_keys.txt" "https://www.dropbox.com/s/yxebmo4gm0qq7nj/test_name_keys.txt?dl=1" || \
    echo "test_name_keys.txt could not be accessed"
    echo "test_name_keys.txt created in $key_loc"
  else
    echo "test_name_keys.txt already exists in $key_loc"
  fi

  if [ ! -f "$key_loc/test_embedding_keys.txt" ] ; then
    curl -Lo "test_embedding_keys.txt" "https://www.dropbox.com/s/kl5s77evy8m9mpm/test_embedding_keys.txt?dl=1" || \
    echo "test_embedding_keys.txt could not be accessed"
    echo "test_embedding_keys.txt created in $key_loc"
  else
    echo "test_embedding_keys.txt already exists in $key_loc"
  fi

}

function mk_keys {
  # Makes key files and directory
  # args: dir (directory at which to create keys)

  dir=$key_dir

  if [ ! -d "$dir" ] ; then
    echo "Creating keys directory at $dir"
    mkdir "$dir"
  else
    echo "Keys directory already exists at $dir"
  fi

  cd "$dir" || echo "Something went wrong: $dir cannot be accessed"

  mk_key_files "$dir"
}

function update_json {
  # Updates json key file
  # args: json_path (path to json key file)

  json_path=$path_to_json

  if [ ! -f "$json_path" ] ; then
    echo "Creating json key file at $json_path"
    touch "$json_path"
    printf '{\n    "names": "%s/test_name_keys.txt",\n    "embeddings": "%s/test_embedding_keys.txt"\n}\n' "$key_dir" "$key_dir" \
    > "$json_path"
  fi

}

# make dirs and files
mk_keys "$key_dir"

# update json key file
update_json "$path_to_json"
