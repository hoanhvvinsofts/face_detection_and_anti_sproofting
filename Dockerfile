# FROM python:3.7.11

# ADD . .

# RUN apt-get update
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN python -m pip install --upgrade pip
# RUN pip install opencv-python
# RUN pip install cmake
# RUN pip install -r requirements.txt

# # Run python script
# CMD ["python", "main.py"]

FROM timekeeping

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
