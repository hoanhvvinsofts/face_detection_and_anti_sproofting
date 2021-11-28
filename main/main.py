from src.increment_data import main
import sys
import logging

logging.basicConfig(filename="logging.txt", level=logging.INFO)
logging.basicConfig(filename="logging.csv", level=logging.INFO)

sys.path.append("src")


# try:
main("E:/Timekeeping/Face Recognition with InsightFace/datasets/videos_input/hoan.mp4", "Hoan")
# except Exception as E:
#     print(E)
#     logging.info(E)
