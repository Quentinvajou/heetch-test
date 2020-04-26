#!/bin/bash

declare -A files_to_dl
files_to_dl=(["1jXKIYSAuA_z88ttr1B-2fETEzARGPsT3"]="drivers" ["1NO7vh221-r63JzHdoTTihVJvKdcFK5HQ"]="bookingRequests" ["1hzFf3ZYfGC2V0o7asKsLzp6-n80EL86k"]="rideRequests")

if [ ! -d "data/raw/" ]
then
  mkdir -p data/raw/
fi
# Iterate the string array using for loop
for val in "${!files_to_dl[@]}"; do
   gdown https://drive.google.com/uc?id=$val -O data/raw/${files_to_dl[$val]}.log
done
