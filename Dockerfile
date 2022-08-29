FROM conda/miniconda3
RUN conda update conda
RUN apt-get -y update
RUN apt-get -y install git
RUN git checkout docker
RUN git clone https://github.com/drduda/TokenMIR.git
WORKDIR /TokenMIR
RUN ls
RUN conda env create -f env.yml
CMD ["bash"]