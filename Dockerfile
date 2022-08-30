FROM conda/miniconda3
RUN conda update conda
RUN apt-get -y update
RUN apt-get -y install git
RUN git clone https://github.com/drduda/TokenMIR.git
WORKDIR /TokenMIR
RUN git checkout docker
RUN ls
RUN conda env create --force -f=env.yml -n base
CMD ["bash"]
