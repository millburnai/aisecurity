
# Unlocks key files by making them writeable

cd "$HOME/PycharmProjects/facial-recognition/" || echo "Permission denied: key files not writeable" && exit
chmod 600 _keys # directory is writeable (and readable) by owner only
chmod -R 600 _keys # key files is writeable (and readable) by owner only