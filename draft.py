import datetime

checkin_time = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
checkin_time = checkin_time.split(" ")
checkin_time_date, checkin_time_time = checkin_time[0], checkin_time[1]
print(checkin_time_date, checkin_time_time)
