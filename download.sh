current_dir=$(pwd)

if [ ! -f "$current_dir/config" ] ; then 
	echo "\033[0;96m~/Downloading config directory\033[0m"
	wget -O "$current_dir/config" \
		"https://www.dropbox.com/sh/d4ho53m2n2u3tmn/AAAlLAcE1pnMshdb5xfBkqjRa?dl=1" \
		|| echo "\033[0;31~/Error downloading config directory\033[0m"
fi