dump (){
        curl -X POST https://content.dropboxapi.com/2/files/upload \
            --header "Authorization: Bearer $1" \
            --header "Dropbox-API-Arg: {\"path\": \"$2\",\"mode\": \"overwrite\",\"autorename\": true,\"mute\": false,\"strict_conflict\": false}" \
            --header "Content-Type: application/octet-stream" \
            --data-binary @$3
}

dump $1 $2 $3
