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
    echo -e "\033[0;95mDownloading test_name_keys.txt\033[0m"
    curl -Lo "test_name_keys.txt" "https://www.dropbox.com/s/yxebmo4gm0qq7nj/test_name_keys.txt?dl=1" || \
    echo -e "\033[0;31mError: could not download test_name_keys.txt\033[0m"
  fi

  if [ ! -f "$key_loc/test_embedding_keys.txt" ] ; then
    echo -e "\033[0;95mDownloading test_embedding_keys.txt\033[0m"
    curl -Lo "test_embedding_keys.txt" "https://www.dropbox.com/s/kl5s77evy8m9mpm/test_embedding_keys.txt?dl=1" || \
    echo -e "\033[0;31mError: could not download test_embedding_keys.txt\033[0m"
  fi

}

function mk_keys {
  # Makes key files and directory
  # args: dir (directory at which to create keys)

  dir=$key_dir

  if [ ! -d "$dir" ] ; then
    echo -e "\033[0;95mCreating keys directory at $dir\033[0m"
    mkdir "$dir"
  fi

  cd "$dir" || echo -e "\033[0;31mError: cannot access $dir\033[0m"

  mk_key_files "$dir"
}

function update_json {
  # Updates json key file
  # args: json_path (path to json key file)

  json_path=$path_to_json

  if [ ! -f "$json_path" ] ; then
    echo -e "\033[0;95mCreating json key file at $json_path\033[0m"
    touch "$json_path"
    printf '{\n    "names": "%s/test_name_keys.txt",\n    "embeddings": "%s/test_embedding_keys.txt"\n}\n' "$key_dir" "$key_dir" \
    > "$json_path"
  fi

}

# make dirs and files
mk_keys "$key_dir"

# update json key file
update_json "$path_to_json"

echo -e "\033[0;96m~/.aisecurity/keys is up to date\033[0m"
