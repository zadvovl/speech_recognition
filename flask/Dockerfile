# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.5.0

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD ./flask /app

# Transfer some app configurations to /app in container
COPY config.json /app
COPY read_config.py /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN pip install -r requirements.txt

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]