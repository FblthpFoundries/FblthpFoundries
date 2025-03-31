#!/bin/sh

while getopts 'berslk' flag; do
	case "${flag}" in
		b) docker build -t fblthp/mse .
			echo "build";;
		e) docker stop mse
			echo "stop";;
		r) docker restart mse
			echo "restart";;
		s) docker start mse
			echo "start";;
		l) docker container ls
			echo "list";;
		k) docker rm mse
			echo "remove";;

	esac
	exit 1
done

docker run -d -p 6969:6969 --name mse fblthp/mse

