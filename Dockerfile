FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential curl
RUN pip3 install --upgrade pip

# Install TA-lib
RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | \
  tar xzvf - && \
  cd ta-lib && \
  sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h && \
  ./configure && make && make install && \
  cd .. && rm -rf ta-lib

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Prepare environment
RUN mkdir /q-trader
# Set the working directory
WORKDIR /q-trader

# Install any needed packages specified in requirements.txt
COPY requirements.txt /q-trader
RUN pip3 install -r ./requirements.txt --no-cache-dir

# Copy the current directory contents into the container
COPY . /q-trader

ENTRYPOINT ["python3"]
# Run app.py when the container launches
CMD ["service.py"]
