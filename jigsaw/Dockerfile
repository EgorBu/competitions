# The underlying base is kaggle/python
FROM kaggle/python

RUN pip install networkx==1.11

EXPOSE 8888

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

ENV PYTHONPATH $PYTHONPATH:/app

# Run python when the container launches
CMD ["python"]
