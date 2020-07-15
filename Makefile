all:
	$(MAKE) linux7
# ================================================================
# --------==== IMPORTANT ====--------
HOST = linux7
HOST_PREFIX = /tmp2/b04303128
PROJECT = CNL-Final
RSYNC_EXCLUDE_LIST = *.git* *__pycache__* *.sqlite3*
RSYNC_EXCLUDE := $(RSYNC_EXCLUDE_LIST:%=--exclude="%")
linux7:
	rsync -avzh ../$(PROJECT)/ $(HOST):$(HOST_PROJECT) $(RSYNC_EXCLUDE)
	ssh linux7 "cd $(HOST_PROJECT) && make django_migrate && make touch_ini"
	# ssh linux7 "cd $(HOST_PROJECT) && make touch_ini"
	# ssh linux7 "cd $(HOST_PROJECT) && make task_tmux TASK=train_sa"
	# ssh linux7 "cd $(HOST_PROJECT) && make task_tmux TASK=predict_sa WINDOW_NAME=human"
	
linux7_full:
	rsync -avzh ../$(PROJECT)/ $(HOST):$(HOST_PROJECT) $(RSYNC_EXCLUDE)

cnlfinal:
	rsync -avzh ../$(PROJECT)/ $@:$(HOST_PROJECT) $(RSYNC_EXCLUDE)
	# ssh linux7 "cd $(HOST_PROJECT) && make django_migrate && make touch_ini"
	# ssh linux7 "cd $(HOST_PROJECT) && make touch_ini"
	# ssh linux7 "cd $(HOST_PROJECT) && make task_tmux TASK=train_sa"
	ssh $@ "cd $(HOST_PROJECT) && echo predict_all"

cnlfinal_homedir:
	mkdir -p homedir
	touch homedir/__init__.py
	rsync -avzh cnlfinal:~/predict.py ./homedir/
	rsync -avzh cnlfinal:~/train.py ./homedir/
	rsync -avzh cnlfinal:~/tail.py ./homedir/
	rsync -avzh cnlfinal:~/tail_packet.cpp ./homedir/
	rsync -avzh cnlfinal:~/sync_packet.sh ./homedir/
	
	
# ================================================================
train_sa:
	mkdir -p model
	$(PYTHON) -m codebox.sa
predict_sa:
	$(PYTHON) -m codebox.sa_predict \
	--model="/tmp2/b04303128/CNL-Final/model/sa8.7___" \
	--mac="fc183c5d8588"
predict_all:
	$(PYTHON) -m codebox.sa_predict \
	--model="/tmp2/b04303128/CNL-Final/model/sa8.7___" --mac=""

WINDOW_NAME = runTask
SESSION_NAME = agent
task_tmux:
	python -m proc.tmux "$(MAKE) $(TASK)" \
	--session=$(SESSION_NAME) \
	--window=$(WINDOW_NAME) \
	--kill --overset

DCODE_SERVER = archive/code-server1.1140-vsc1.33.1-darwin-x64/code-server
LCODE_SERVER = archive/code-server1.1140-vsc1.33.1-linux-x64/code-server
CODE_SERVER := $(LCODE_SERVER)
dcode:
	$(DCODE_SERVER) --help
	# -ln -s /Users/qtwu/anaconda/pkgs/openssl-1.0.2r-h1de35cc_0/ssl /Users/qtwu/anaconda/ssl
	# $(DCODE_SERVER) --cert=./aiweb/mycert.pem --cert-key=./aiweb/mykey.key

lcode:
	$(CODE_SERVER) --cert=./aiweb/mycert.pem --cert-key=./aiweb/mykey.key

server_start:
	cd aiweb && $(UWSGI) --ini="uwsgi.ini"
touch_ini:
	touch ./aiweb/uwsgi.ini
	$(MAKE) setup_tmux
django_migrate:
	export PYTHONPATH=py368
	cd aiweb && \
	$(PYTHON) manage.py makemigrations && \
	$(PYTHON) manage.py makemigrations AIWeb && \
	$(PYTHON) manage.py migrate
django_ipython:
	cd aiweb && $(PYTHON) manage.py shell
django_super:
	cd aiweb && $(PYTHON) manage.py createsuperuser

train_sa_macbook django_migrate_macbook django_ipython_macbook django_super_macbook: %_macbook:
	$(MAKE) $* PYTHON=python

ipython:
	$(MINI_BIN)/ipython

HOST_PROJECT := $(HOST_PREFIX)/$(PROJECT)/
MINI_DIR = $(HOST_PREFIX)/mini
MINI_BIN = $(HOST_PREFIX)/mini/bin
CONDA := $(MINI_BIN)/conda
PIP := $(MINI_BIN)/pip
PYTHON := $(MINI_BIN)/python
UWSGI := $(MINI_BIN)/uwsgi


MINI_URL = https://repo.continuum.io/miniconda/Miniconda3-4.3.30-Linux-x86_64.sh
MINI_SH = $(HOST_PREFIX)/mini.sh
install_py368:
	mkdir -p $(HOST_PREFIX)/mini
	wget $(MINI_URL) -O $(MINI_SH)
	sh $(MINI_SH) -b -u -p $(MINI_DIR)
	$(CONDA) update conda -y

install_uwsgi:
	$(CONDA) config --add channels conda-forge
	$(CONDA) install --yes uwsgi

install_django:
	$(CONDA) install --yes mako
	$(CONDA) install --yes django
	$(PIP) install django-sslserver

install_channel:
	$(CONDA) install --yes twisted
	$(PIP) install -U channels

install_torch:
	$(PIP) install pandas
	$(PIP) install torch

setup_tmux:
	tmux set-option -g default-shell /bin/zsh
	# tmux set-option -g mouse off
	tmux set-option -g remain-on-exit off
	tmux set -g base-index 1
	tmux setw -g pane-base-index 1
	tmux bind m set -g mouse on \; display "Mouse ON"
	tmux bind M set -g mouse off \; display "Mouse OFF"
	tmux set-option -g allow-rename off

download_packet:
	mkdir -p data
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet1.csv ./data/
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet2.csv ./data/
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet3.csv ./data/
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet4.csv ./data/
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet5.csv ./data/
	rsync -avzh cnlfinal:/home/cnl/labeled_packets/labeled_packet6.csv ./data/

download_model:
	mkdir -p model
	rsync -avzh linux7:/tmp2/b04303128/CNL-Final/model/sa8.7___ ./model/