server_path="cnl@linux6.csie.ntu.edu.tw:~/packet/packet$1"

while :
do
	rsync --append -z -e 'ssh -p 21022' /home/pi/packets $server_path 
	sleep 1 # Wait
done
