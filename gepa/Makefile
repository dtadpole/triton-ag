# CUDA_VISIBLE_DEVICES = ${GPU}
ENV_VARS ?= PYTHONNOUSERSITE=1 PYTHONPATH=${PYTHONPATH}:${PWD}
HOST=$(shell hostname)
IS_DEVSERVER=$(shell hostname | grep -E -c "dev.*\.facebook\.com")
META_PROXY := https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 no_proxy='\''\'\'''\''.facebook.com|.tfbnw.net|*.fb.com'\''\'\'
VLLM_SETTING := VLLM_ALLOW_RUNTIME_LORA_UPDATING=True HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0

help:
	@echo "Available targets:"
	@echo "  build_docker    - build docker image"

host_check:
	@echo "Current hostname is "${HOST}
	@echo "Is it meta devserver 1:yes; 0:no ?  Ans: "${IS_DEVSERVER}

build_docker: Dockerfile
ifeq (${IS_DEVSERVER}, 1)
	$(META_PROXY) docker build --network=host --progress=plain  -t gepa .
else
	docker build --network=host --progress=plain  -t gepa .
endif

wandb_login:
	wandb login --host=http://devgpu139.cco2.facebook.com:8010/

mount_shared_drive:
	echo 'dummy' | sshfs -o password_stdin -p 8081 codegen@devvm8492.cco0.facebook.com:/shared/ shared/


env_gepa_start:
	$(META_PROXY) docker run -d \
		--name gepa \
		--replace \
		--gpus all \
		--cap-add SYS_ADMIN \
		--net=host \
		--shm-size=128g \
		--pids-limit -1 \
		--ulimit nofile=65536:65536 \
		--ulimit nproc=-1:-1\
		--ulimit memlock=-1:-1 \
		-v ~/.ssh/:/root/.ssh \
		-v ~/.bashrc:/root/.bashrc \
		-v ~/.netrc:/root/.netrc \
		-v ~/.gitconfig:/root/.gitconfig \
		-v ~/.keys/:/root/.keys/ \
		-v /data/users/${USER}/:/root/.cache/ \
		-v ${PWD}:/workspace/ \
		--cap-add SYS_ADMIN --device /dev/fuse \
		--security-opt apparmor:unconfined \
		--privileged \
		localhost/gepa \
		/bin/bash -c "make mount_shared_drive && tail -f /dev/null"

env_gepa:
	docker exec -it gepa /bin/bash
