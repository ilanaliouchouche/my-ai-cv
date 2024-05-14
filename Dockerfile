FROM python:3.11

# Set the working directory
RUN mkdir -p /usr/src/assistant
WORKDIR /usr/src/assistant

# Copy the current directory contents into the container at /usr/src/app
COPY ./assistant/chromadb /usr/src/assistant/chromadb
COPY ./assistant/requirements.txt /usr/src/assistant/requirements.txt
COPY ./assistant/app.py /usr/src/assistant/app.py
COPY ./assistant/models /usr/src/assistant/models
COPY ./assistant/.env /usr/src/assistant/.env


RUN pip install --no-cache-dir -r /usr/src/assistant/requirements.txt

# Expose the port the app runs on
EXPOSE 7860

# Run app.py when the container launches
CMD ["python", "app.py"]