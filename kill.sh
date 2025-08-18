ps aux | grep 'vizwiz' | awk '{print $2}' | xargs kill
