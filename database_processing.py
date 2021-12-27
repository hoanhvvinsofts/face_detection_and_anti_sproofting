
import configparser
import os
import datetime
import sqlite3

# Init config file
config = configparser.ConfigParser()
config.read("config.ini")
train_dataset_path = config["DATASET"]["train_dataset_path"]
database = config["DATABASE"]["database_name"]
database_file = config["DATABASE"]["database_file"]
conn = sqlite3.connect(database_file)
conn.row_factory = sqlite3.Row
c = conn.cursor()


def post_process_check_time(name, checkin_time):
    checkin_time = checkin_time.split(" ")
    checkin_time_date, checkin_time_time = checkin_time[0], checkin_time[1]
    name = name

    return name, checkin_time_date, checkin_time_time

def insert_database(employee, database=database):
    with conn:
        c.execute("INSERT INTO " + str(database) + " VALUES (:name, :day, :checkin, :checkout, :time, :total_time)", 
                  {"name":employee.name, "day":employee.day, "checkin":employee.checkin, "checkout":employee.checkout, 
                   "time":employee.time, "total_time":employee.total_time})

def insert_database_none_values_except_name(name, today_day="now", database=database):
    if today_day == "now":
        today_day = datetime.datetime.now().strftime('%d-%m-%Y')
        with conn:
            c.execute("INSERT INTO " + str(database) + " VALUES (:name, :day, :checkin, :checkout, :time, :total_time)", 
                    {"name":name, "day":today_day, "checkin":"", "checkout":"", "time":0.0, "total_time":0.0})

def update_time(employee_name, day, check_time, database=database):
    c.execute("""SELECT * FROM {} WHERE name=:name AND day=:day
              AND checkin=:checkin""".format(database), 
              {"name":employee_name, "day":day, "checkin":check_time})
    result = c.fetchone()
    if result != None:
        check_time = result["checkin"] + ", " + check_time
        with conn:
            c.execute("""
                        UPDATE {} SET checkin = :checkin
                        WHERE name = :name AND day = :day
                        """.format(database),
                        {"name": employee_name, "day": day, "checkin": check_time})
    else:
        with conn:
            c.execute("""
                    UPDATE {} SET checkin = :checkin
                    WHERE name = :name AND day = :day
                    """.format(database),
                    {"name": employee_name, "day": day, "checkin": check_time})

def show_table_database(database=database):
    with conn:
        c.execute("SELECT * FROM " + str(database) + ";")
        # results = c.fetchall()
        results = [dict(row) for row in c.fetchall()]
        for result in results:
            print(result)

def if_new_days(checkin_time_date, database=database):
    c.execute("SELECT * FROM {} WHERE day=:day".format(database), {"day":checkin_time_date})
    result = c.fetchall()
    if result is not None:
        for label in os.listdir(train_dataset_path):
            insert_database_none_values_except_name(label)
    
def database_processing(name, checkin_time):
    name, checkin_time_date, checkin_time_time = post_process_check_time(name, checkin_time)
    c.execute("""
            CREATE TABLE IF NOT EXISTS {}(
                name text,
                day text,
                checkin text,
                checkout text,
                time real,
                total_time real
            )
            """.format(database))

    # Process if recognize day is go to the new day, insert all emply values
    if_new_days(checkin_time_date)
    
    # Update checkin_time when recognize new face
    update_time(name, checkin_time_date, checkin_time_time)
