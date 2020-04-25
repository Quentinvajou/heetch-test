CREATE DATABASE IF NOT EXISTS heetch_raw


CREATE TABLE bookingRequests
(
  request_id character varying(50) NOT NULL,
  logged_at TIMESTAMP,
  ride_id character varying(50),
  driver_id character varying(50),
  driver_accepted character(5),
  driver_lat float,
  driver_lon float
  )

LOAD DATA INFILE '/var/lib/mysql/data/raw/bookingRequests.log'
INTO TABLE bookingRequests
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(request_id, @var1, ride_id, driver_id, @var2, driver_lat, driver_lon)
SET logged_at=FROM_UNIXTIME(@var1)
;

