#!/usr/bin/env bash

cd "$HOME/PycharmProjects/facial-recognition/security/_keys/" || cd "$HOME/Desktop/facial-recognition/security/_keys/" \
|| echo "Key files not found" && exit
chmod 400 . # nobody can access the _keys directory or _keys/<whatever file>
cd ..