cd server/node-server
gnome-terminal -x bash -c "node ./server.js"
cd ../../front-end
gnome-terminal -x bash -c "npm run dev"
cd ../
gnome-terminal -x bash -c "sudo ffserver -f ./ffmpeg/server.conf"
