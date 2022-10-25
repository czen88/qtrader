FROM python:3.8.0b1-stretch
RUN apt-get update -y
RUN pip install --upgrade pip

# Required for TA-LIB
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xzvf - \
  && cd ta-lib \
  && sed -i "s|0.00000001|0.000000000000000001 |g" src/ta_func/ta_utility.h \
  && ./configure && make && make install \
  && cd .. && rm -rf ta-lib

# Prepare environment
RUN mkdir /q-trader
# Set the working directory
WORKDIR /q-trader

# Install any needed packages specified in requirements.txt
COPY requirements.txt /q-trader
RUN pip install -r requirements.txt --no-cache-dir

# Copy the current directory contents into the container
COPY . /q-trader

# Run bot when the container launches
CMD ["python", "bot.py"]
