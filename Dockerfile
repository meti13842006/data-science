FROM tensorflow/tensorflow:latest-jupyter

RUN mkdir -p /program

WORKDIR /program
COPY requirements.txt /program/requirements.txt

RUN pip install -r requirements.txt

COPY app.py /program/app.py

CMD ["python", "/program/app.py"]