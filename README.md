# Face Detection and Face Anti Sproofing

## Requirements (all with pip)
```
requirements.txt
```

## Test with mediapipe detector
```
cd main
python main.py
```

## Usage
run main.py
Add a new Face by press "a" button
When recognize a face, name label and current time will save to timekeeping.db sqlite3 database

## Database (SQLITE3)
name: text,			(label name in training data)
day: text,			(%day-%month-%Year)
checkin: text,		(%Hour:%Minute:%Second)
checkout: text,		(not in this version)
time: real,			(not in this version)
total_time: real	(not in this version)

## Configuration
All config is store in config.ini
Create py run create_config.py
