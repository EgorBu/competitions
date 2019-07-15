FROM kaggle/python

CMD ["bash"]

RUN git clone --recursive https://github.com/EgorBu/wmd-relax && cd wmd-relax &&  \
    pip3 install -e .&& pip3 install spacy &&  python3 -m spacy download en_core_web_lg

WORKDIR /code

COPY . /code

CMD ["bash"]