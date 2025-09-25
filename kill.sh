ps aux | grep 'llamavid' | awk '{print $2}' | xargs kill
