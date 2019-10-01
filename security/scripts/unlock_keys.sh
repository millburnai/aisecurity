#!/usr/bin/env bash

cd "$HOME/PycharmProjects/facial-recognition/security/" || cd "$HOME/Desktop/facial-recognition/security/" \
|| echo "Key files not found" && exit
chmod 775 _keys/ # everyone can read, write, and excute
chmod -R 775 _keys/