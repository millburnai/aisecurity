#!/usr/bin/env bash

# "aisecurity.create_config"
# Program to create config file (~/.aisecurity.json)

function confirm {
  # Confirm script continuation
  # args: message (prompt)

  message=$1

  read -rp "$message" confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

}

if [ -d "$HOME/.aisecurity/aisecurity.json" ] ; then
  confirm "$HOME/.aisecurity/aisecurity.json already exists. Overwrite? (y/n):"
fi

cd .
