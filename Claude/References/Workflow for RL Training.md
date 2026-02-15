# ***Workflow for RL Training***

## ***Components***

## ***Organization of files***

|  | Source Code | Config File | Data |
| :---: | :---: | :---: | :---: |
| Workflow | workflowClient.py workflowServer.py  | workflow.yaml {source\_dir}/workflow/\*  | \~/.workflow/\*  |
| Inference | inferenceComposer.pyinferenceClient.py  | inferenceClient.yaml {source\_dir}/inference/\*  | \~/.inference/\*\*  |
| KbEval | kbEvalServer.py kbEvalClient.py | kbEval.yaml | \~/.kbeval |
| Training | trainerBase.py trainerGRPO.py trainerRFT.py trainerMain.py  | trainerBase.yaml trainerGRPO.yaml trainerRFT.yaml {source\_dir}/trainer/\*  | \~/.trainer/\*\*  |
| Data Movements | workflowRsync.py  | workflowRsync.yaml  | n.a.  |

## ***Notes before start the full workflow***

* I use tmux intensively in the example, you don’t have to, but I find it easy to run and monitor the running code. Here is a list of tmux aliases used ([bookmark](#bookmark=kix.jxyek6w3espl)).  
* Run inference ([bookmark](#bookmark=kix.xtg73760x5yl)) and trainer ([bookmark](#bookmark=kix.jcy6qaxrzjf5)) standalone first, this will verify the setup is correct, before starting the workflow  
* Setup the vLLM server ([bookmark](#bookmark=kix.7mv5b2x1dnjy))  
* Setup the kbEval server ([bookmark](#bookmark=kix.p14r70ew2po3))  
* Make the following folders in the home directory and mount them in the container environment   
  * mkdir \~/.workflow  
  * mkdir \~/.trainer  
  * mkdir \~/.inference  
  * mkdir \~/.keys  
  * mkdir \~/.cache  
  * mkdir \~/.kbeval  
  * Example docker run command  
    * docker run \-it  \--gpus all \--net=host \-v \~/.workflow/:/root/.workflow/ \-v \~/.trainer/:/root/.trainer/ \-v \~/.inference/:/root/.inference/ \-v \~/.bashrc:/root/.bashrc \-v \~/.netrc:/root/.netrc \-v \~/.gitconfig:/root/.gitconfig \-v \~/.keys/:/root/.keys/ \-v /data/users/${USER}/:/root/.cache/ \-v \~/.kbeval:/root/.kbeval/ \-v ${PWD}:/workspace/ localhost/triton\_ag /bin/bash \-c "make wandb\_login && /bin/bash"

## ***Workflow***

### ***Prepare workflow.yaml***

In the workflow.yaml file, and add your own workflow in the registry section:

| registry:   TC\_0.1.0\_14B.m:     short\_name: "t14m"     config\_path: "workflow/14B.m.yaml"     data\_dir: "\~/.workflow"   TC\_0.1.0\_14B.n:     short\_name: "t14n"     config\_path: "workflow/14B.n.yaml"     data\_dir: "\~/.workflow"   my\_tag.a:                                          \# ⇐ newly added     short\_name: “ma”                           \# ⇐ short name for debugging     config\_path: "workflow/my.a.yaml”   \# ⇐ make a copy of “14B.n.yaml”     data\_dir: “\~/.workflow”                    \# ⇐ state of workflow is stored, prefix is auto added  |
| :---- |

Make a copy of 14B.n.yaml as \[workflow/my.a.yaml\], check \[workflow/my.a.yaml\], edit config to your needs:

| global:   prefix\_tag: "my\_tag.a"                \# ⇐ this is your workflow tag   start\_epoch: 0                            \# ⇐ edit these to your needs   start\_block: 0                             \# ⇐ edit these to your needs   end\_epoch: 20                          \# ⇐ edit these to your needs   end\_block: 24                           \# ⇐ edit these to your needs |
| :---- |

### ***Start workflow server***

| % tn workflow % while true; do time python ./workflowServer.py \--host :: \--port 8488; sleep 5; done % ctrl-b, d |
| :---- |

## ***Inference***

### ***Run Inference Standalone***

Run inference against vLLM server (has logprobs, has token\_id, can be used for both finetuning, and GRPO)  
We need to launch a kbeval server, and a vllm server supporting log-probs before running the following command. 

| % python ./inferenceComposer.py \--prefix\_tag my\_tag.a \--input\_dir kernel\_bench/level1/ \--provider local \--model qwen3-14b |
| :---- |

Run inference against any 3pd LLM server (no logprobs, can be used for finetuning, but not GRPO)

| % python ./inferenceComposer.py \--prefix\_tag my\_tag.a \--input\_dir kernel\_bench/level1/ \--module\_file inference/codeGenEval.module.chat.yaml \--provider deepseek \--model deepseek-chat |
| :---- |

### ***Run Inference with a workflow***

Run multiple workers, worker 01 below:

| % tn codeGen1 % while true; do python ./inferenceComposer.py \--prefix\_tag cudacoder\_eval\_one\_turn.a \--use\_global\_queue codeGenEval.base \--proc\_id 01; sleep 5; done % ctrl-b, d |
| :---- |

Run multiple workers, worker 02 below:

| % tn codeGen2 % while true; do python ./inferenceComposer.py \--prefix\_tag cudacoder\_eval\_one\_turn.a \--use\_global\_queue codeGenEval.base \--proc\_id 02; sleep 5; done % ctrl-b, d |
| :---- |

Log Probability Server

Logps\_server:  
logp1  
while true; do CUDA\_VISIBLE\_DEVICES=1 python inferenceCustomServer.py \--prefix\_tag cudacoder\_gspo\_qwen32b.t01; sleep 1; done  
logp2  
while true; do CUDA\_VISIBLE\_DEVICES=2 python inferenceCustomServer.py \--prefix\_tag cudacoder\_gspo\_qwen32b.t01 \--port 8003; sleep 1; done

## ***Training***

### ***Run Trainer Standalone***

(These assume “my\_tag.a\_000\_00” has been generated by standalone inference above, check the folder \~/.inference/output/ for the filename)

| % python ./trainerGRPO.py \--input\_dir \~/.inference/output/ \--input\_tag my\_tag.a\_20250815\_201330 \# Modify the input\_tag based on the file generated from inferenceComposer in the .inference/output/ folder |
| :---- |

If not, duckdb.duckdb.IOException: IO Error: No files found that match the pattern "/root/.inference/codeGenEval/my\_tag.a\_000\_00/\*\*/\*\_generated\_eval.json"

| % python ./trainerSFT.py \--input\_dir \~/.inference/output \--input\_tag my\_tag.a\_20250813\_034658 |
| :---- |

### ***Run Trainer with a workflow***

| % tn trainer % while true; do CUDA\_VISIBLE\_DEVICES=0 python trainerMain.py \--prefix\_tag cudacoder\_gspo\_qwen32b.t10; sleep 5; done % ctrl-b, d |
| :---- |

## ***Data Movements & Misc***

### ***Initialize Workflow***

| % python ./[workflowInit.py](http://workflowInit.py) \--prefix\_tag my\_tag.a |
| :---- |

### ***Sync LoRA Adapter***

| % tn sync % while true; do time python ./workflowSync.py \--prefix\_tag cudacoder\_gspo\_qwen32b.t01 \--module\_file workflow/sync.module.vllm+logp\_norsync.yaml; sleep 5; done % ctrl-b, d |
| :---- |

# ***Appendix***

## ***Tmux***

Tmux is a tool that can run code in the background, at the same time, have a direct view on the program output.  Below is a list of aliases I use:

alias t='tmux'  
alias tl='tmux list-sessions'  
alias tn='tmux new-session \-s'  
alias td='tmux detach'  
alias ta='tmux attach \-t'  
alias tk='tmux kill-session \-t'  
alias tr='tmux rename-session \-t'  
alias tlw='tmux list-windows'  
alias tnw='tmux new-window \-n'  
alias tkw='tmux kill-window \-t'  
alias trw='tmux rename-window'

## ***vLLM server***

Start podman:

| % sudo dnf install \-y nvidia-container-toolkit % sudo nvidia-ctk cdi generate \--output=/etc/cdi/nvidia.yaml % nvidia-ctk cdi list % tn vllm % podman/run-vllm.sh  |
| :---- |

/root/run.sh

| \#\!/bin/bash pwd cd $HOME source .venv/bin/activate export VLLM\_ALLOW\_RUNTIME\_LORA\_UPDATING=True export HF\_HUB\_DISABLE\_XET=1 export HF\_HUB\_ENABLE\_HF\_TRANSFER=0 cd \- pwd "$@" |
| :---- |

Run vLLM inside docker:

| \# /root/run.sh python \-m vllm.entrypoints.openai.api\_server \--model Qwen/Qwen3-14B \--port 8091 \--host 0.0.0.0 \--api-key dummy \--enable-lora \--max-lora-rank 128 \--max-loras 8 \--gpu-memory-utilization 0.9 \--max\_model\_len 24576 \--load\_format safetensors \--guided\_decoding\_backend guidance \--guided-decoding-disable-fallback \--enable\_auto\_tool\_choice \--tool\_call\_parser hermes \--scheduling\_policy priority \--enable\_chunked\_prefill \--max\_num\_batched\_tokens 8192 \--max\_num\_seqs 16 \--max\_log\_len 0 \--trust\_remote\_code \--enable\_prefix\_caching \--prefix-caching-hash-algo sha256 \--generation-config vllm \--override-generation-config '{"temperature":0.6,"top\_p":1.0,"top\_k":0,"repetition\_penalty":1.0}' \--return-tokens-as-token-ids \--enforce-eager  \# ctrl-b, d |
| :---- |

## ***kbEvalServer***

| % tn kbEval % while true; do time python ./[kbEvalServer.py](http://kbEvalServer.py) \--local\_host \--port 5676 \--device 7; sleep 1; done |
| :---- |

### ***Restart kbEval server every {TIMEOUT} secs***

| % crontab \-e |
| :---- |

Install crontab that restarts kbEvalServer

| \*/2 \* \* \* \* /home/centos/triton-ag/kbEvalRestart.sh |
| :---- |

How to get wandb account at Meta

1. Follow this [wiki](https://www.internalfb.com/wiki/AI_Music/Archived/Setting_up_WandB/) step by step  
2. In the docker container with wandb installed.   
   1. wandb login \--relogin \--host=[https://fairwandb.org](https://fairwandb.org)  
   2. Add mount “-v \~/.netrc:/root/.netrc” in the docker run command  
3. Setting project access as restricted. (need upgrade)  
   1. This function is not supported, and won’t be supported as meta will migrate to other platform in the future


4. A workaround: set up a team, and add all the team members as admin. [https://fairwandb.org/code-gen/projects](https://fairwandb.org/code-gen/projects)  
5. For migrating existing reports to new team using APIs (Those APIs don’t work well, it is better to log into the team account directly)  
   1. [https://docs.wandb.ai/ref/python/public-api/api/\#reports](https://docs.wandb.ai/ref/python/public-api/api/#reports)  
   2. [https://docs.wandb.ai/guides/reports/edit-a-report/](https://docs.wandb.ai/guides/reports/edit-a-report/)  
   3. [https://docs.wandb.ai/guides/reports/create-a-report/](https://docs.wandb.ai/guides/reports/create-a-report/)

