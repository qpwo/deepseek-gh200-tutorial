# Serving deepseek v3 and r1 on GH200s

### Create instances

Make [one instance](https://cloud.lambdalabs.com/instances) and note the name of your ssh key and your shared filesystem.

```sh
export ssh_key_name=my-key
export shared_fs_name=shared
```

We'll use the API to make the rest. You'll need an [api key](https://cloud.lambdalabs.com/api-keys).

```sh
sudo apt-get install -y jq

export LAMBDA_API_KEY="..."

function lambda-api {
    local method=$1
    local route=$2
    shift 2
    curl --fail -q -X "$method" -H "Authorization: Bearer $LAMBDA_API_KEY" "https://cloud.lambdalabs.com/api/v1/$route" -H "Content-Type: application/json" "$@"
}
export -f lambda-api

num_want=16
num_got=1
while [[ $num_got -lt $num_want ]]; do
    lambda-api POST instance-operations/launch -d '{
        "region_name": "us-east-3",
        "instance_type_name": "gpu_1x_gh200",
        "ssh_key_names": ["'$ssh_key_name'"],
        "file_system_names": ["'$shared_fs_name'"],
        "quantity": 1,
        "name": "node_'$num_got'"
    }' && ((num_got++))
    echo num_got=$num_got
    sleep 3
done
```

Lambda labs has half-off GH200s right now to get more people used to the ARM tooling. This means you can maybe actually afford to run the biggest open-source models! The only caveat is that you'll have to occasionally build something from source. Here's how I got llama 405b running with full precision on the GH200s.

### Create instances

Llama 405b is about 750GB so you want about 10 96GB GPUS to run it. (The GH200 has pretty good CPU-GPU memory swapping speed -- that's kind of the whole point of the GH200 -- so you can use as few as 3. Time-per-token will be terrible, but total throughput is acceptable, if you're doing batch-processing.) Sign in to lambda labs and create a bunch of GH200 instances. **Make sure to give them all the same shared network filesystem.**

![lambda labs screenshot](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/sjhtn33f1abnha6ng50c.png)

Save the ip addresses to ~/ips.txt.

### Bulk ssh connection helpers

I prefer direct bash & ssh over anything fancy like kubernetes or slurm. It's manageable with some helpers.

```sh
# skip fingerprint confirmation
for ip in $(cat ~/ips.txt); do
    echo "doing $ip"
    ssh-keyscan $ip >> ~/.ssh/known_hosts
done

function run_ip() {
    ssh -i ~/.ssh/lambda_id_ed25519 ubuntu@$ip -- stdbuf -oL -eL bash -l -c "$(printf "%q" "$*")" < /dev/null
}
function run_k() { ip=$(sed -n "$k"p ~/ips.txt) run_ip "$@"; }
function runhead() { ip="$(head -n1 ~/ips.txt)" run_ip "$@"; }

function run_ips() {
    for ip in $ips; do
        ip=$ip run_ip "$@" |& sed "s/^/$ip\t /" &
        # pids="$pids $!"
    done
    wait &> /dev/null
}
function runall() { ips="$(cat ~/ips.txt)" run_ips "$@"; }
function runrest() { ips="$(tail -n+2 ~/ips.txt)" run_ips "$@"; }

function ssh_k() {
    ip=$(sed -n "$k"p ~/ips.txt)
    ssh -i ~/.ssh/lambda_id_ed25519 ubuntu@$ip
}
alias ssh_head='k=1 ssh_k'

function killall() {
    pkill -ife '.ssh/lambda_id_ed25519'
    sleep 1
    pkill -ife -9 '.ssh/lambda_id_ed25519'
    while [[ -n "$(jobs -p)" ]]; do fg || true; done
}
```

### Set up NFS cache

We'll be putting the python environment and the model weights in the NFS. It will load much faster if we cache it.

```sh
# First, check the NFS works.
# runall ln -s my_other_fs_name shared
runhead 'echo world > shared/hello'
runall cat shared/hello


# Install and enable cachefilesd
runall sudo apt-get update
runall sudo apt-get install -y cachefilesd
runall "echo '
RUN=yes
CACHE_TAG=mycache
CACHE_BACKEND=Path=/var/cache/fscache
CACHEFS_RECLAIM=0
' | sudo tee -a /etc/default/cachefilesd"
runall sudo systemctl restart cachefilesd
runall 'sudo journalctl -u cachefilesd | tail -n2'

# Set the "fsc" option on the NFS mount
runhead cat /etc/fstab # should have mount to ~/shared
runall cp /etc/fstab etc-fstab-bak.txt
runall sudo sed -i 's/,proto=tcp,/,proto=tcp,fsc,/g' /etc/fstab
runall cat /etc/fstab

# Remount
runall sudo umount /home/ubuntu/shared
runall sudo mount /home/ubuntu/shared
runall cat /proc/fs/nfsfs/volumes # FSC column should say "yes"

# Test cache speedup
runhead dd if=/dev/urandom of=shared/bigfile bs=1M count=8192
runall dd if=shared/bigfile of=/dev/null bs=1M # First one takes 8 seconds
runall dd if=shared/bigfile of=/dev/null bs=1M # Seond takes 0.6 seconds
```

### Create conda environment

Instead of carefully doing the exact same commands on every machine, we can use a conda environment in the NFS and just control it with the head node.

```sh
# We'll also use a shared script instead of changing ~/.profile directly.
# Easier to fix mistakes that way.
runhead 'echo ". /opt/miniconda/etc/profile.d/conda.sh" >> shared/common.sh'
runall 'echo "source /home/ubuntu/shared/common.sh" >> ~/.profile'
runall which conda

# Create the environment
runhead 'conda create --prefix ~/shared/311 -y python=3.11'
runhead '~/shared/311/bin/python --version' # double-check that it is executable
runhead 'echo "conda activate ~/shared/311c" >> shared/common.sh'
runall 'which python; which pip'
```

```sh
runall ln -s /home/ubuntu/shared/311c  /home/ubuntu/311c
runall 'echo $(ls ~/311c)'

# OLD=/home/ubuntu/311c
# NEW=/home/ubuntu/shared/311c
# sed  -i.bak "s:envs/$OLD/bin:envs/$NEW/bin:" $NEW/bin/*
```

### Install aphrodite dependencies

Aphrodite is a fork of vllm that starts a bit quicker and has some extra features.
It will run the openai-compatible inference API and the model itself.

You need torch, triton, and flash-attention.
You can get aarch64 torch builds from pytorch.org (you do not want to build it yourself).
The other two you can either build yourself or use the wheel I made.

If you build from source, then you can save a bit of time by running the `python setup.py bdist_wheel` for triton, flash-attention, and aphrodite in parallel on three different machines. Or you can do them one-by-one on the same machine.

```sh
runhead pip install 'numpy<2' torch==2.4.0 --index-url 'https://download.pytorch.org/whl/cu124'

# fix for "libstdc++.so.6: version `GLIBCXX_3.4.30' not found" error:
runhead conda install -y -c conda-forge libstdcxx-ng=12

runhead python -c 'import torch; print(torch.tensor(2).cuda() + 2, "torch ok")'
```

#### triton & flash attention from wheels

```sh
runhead pip install 'https://github.com/qpwo/lambda-gh200-llama-405b-tutorial/releases/download/v0.1/triton-3.2.0+git755d4164-cp311-cp311-linux_aarch64.whl'
runhead pip install 'https://github.com/qpwo/lambda-gh200-llama-405b-tutorial/releases/download/v0.1/aphrodite_flash_attn-2.6.1.post2-cp311-cp311-linux_aarch64.whl'

```

#### triton from source

```sh
k=1 ssh_k # ssh into first machine

pip install -U pip setuptools wheel ninja cmake setuptools_scm
git config --global feature.manyFiles true # faster clones
git clone https://github.com/triton-lang/triton.git ~/shared/triton
cd ~/shared/triton/python
git checkout 755d4164 # <-- optional, tested versions
# Note that ninja already parallelizes everything to the extent possible,
# so no sense trying to change the cmake flags or anything.
python setup.py bdist_wheel
pip install --no-deps dist/*.whl # good idea to download this too for later
python -c 'import triton; print("triton ok")'
```

#### flash-attention from source

```sh
k=2 ssh_k # go into second machine

git clone https://github.com/AlpinDale/flash-attention  ~/shared/flash-attention
cd ~/shared/flash-attention
python setup.py bdist_wheel
pip install --no-deps dist/*.whl
python -c 'import aphrodite_flash_attn; import aphrodite_flash_attn_2_cuda; print("flash attn ok")'
```

### Install aphrodite

You can use my wheel or build it yourself.

#### aphrodite from wheel

```sh
runhead pip install 'https://github.com/qpwo/lambda-gh200-llama-405b-tutorial/releases/download/v0.1/aphrodite_engine-0.6.4.post1-cp311-cp311-linux_aarch64.whl'
```

#### aphrodite from source

```sh
k=3 ssh_k # building this on the third machine

git clone https://github.com/PygmalionAI/aphrodite-engine.git ~/shared/aphrodite-engine
cd ~/shared/aphrodite-engine
pip install protobuf==3.20.2 ninja msgspec coloredlogs portalocker pytimeparse  -r requirements-common.txt
python setup.py bdist_wheel
pip install --no-deps dist/*.whl
```

### Check all installs succeeded

```sh
function runallpyc() { runall "python -c $(printf %q "$*")"; }
runallpyc 'import torch; print(torch.tensor(5).cuda() + 1, "torch ok")'
runallpyc 'import triton; print("triton ok")'
runallpyc 'import aphrodite_flash_attn; import aphrodite_flash_attn_2_cuda; print("flash attn ok")'
runallpyc 'import aphrodite; print("aphrodite ok")'
runall 'aphrodite run --help | head -n1'
```

### Download the weights

Go to https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct and make sure you have the right permissions. The approval usually takes about an hour. Get a token from https://huggingface.co/settings/tokens

```sh
runall which python
runhead pip install hf_transfer 'huggingface_hub[hf_transfer]'

runall git config --global credential.helper store
runall huggingface-cli login --token $new_hf

# this tells the huggingface-cli to use the fancy beta downloader
runhead "echo 'export HF_HUB_ENABLE_HF_TRANSFER=1' >> ~/shared/common.sh"
runall 'echo $HF_HUB_ENABLE_HF_TRANSFER'

runall pkill -ife huggingface-cli # kill any stragglers

for k in $(seq 1 10); do
k=$k run_k 'bash shared/git/download.sh' &
done
wait

# we can speed up the model download by having each server download part
local_dir=/home/ubuntu/shared/hff/405b-instruct
for digit in $(seq 1 10); do
k=1 run_k huggingface-cli download --max-workers=32 --revision="main" --include="[0-4]?-of-00191.safetensors" --local-dir=$local_dir meta-llama/Meta-Llama-3.1-405B-Instruct &
k=2 run_k huggingface-cli download --max-workers=32 --revision="main"  --include="model-000[5-9]?-of-00191.safetensors" --local-dir=$local_dir meta-llama/Meta-Llama-3.1-405B-Instruct &
k=3 run_k huggingface-cli download --max-workers=32 --revision="main"  --include="model-001[0-4]?-of-00191.safetensors" --local-dir=$local_dir meta-llama/Meta-Llama-3.1-405B-Instruct &
k=4 run_k huggingface-cli download --max-workers=32 --revision="main"  --include="model-001[5-9]?-of-00191.safetensors" --local-dir=$local_dir meta-llama/Meta-Llama-3.1-405B-Instruct &

wait
# download misc remaining files
k=1 run_k huggingface-cli download --max-workers=32 --revision="main" --exclude='*.pth' --local-dir=$local_dir meta-llama/Meta-Llama-3.1-405B-Instruct
```

### deepseek

```sh
runhead 'ip addr show | grep "inet " | grep -v 127.0.0.1 | awk "{print $2}" | cut -d/ -f1 | head -n 1'
headip=172.26.132.69
runall curl $headip
# node 1

for k in $(seq 1 8); do
k=$k run_k "python -m sglang.launch_server --model-path /home/ubuntu/deepseekv3 --tp 8 --dist-init-addr $headip:5000 --nnodes 8 --node-rank $((k - 1)) --trust-remote-code" &
done

# python -m sglang.launch_server --model-path /home/ubuntu/deepseekv3 --tp 10 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code

runall pkill -ife python
# separate tab:
while true; do
rm page.txt; touch page.txt
runall 'nvidia-smi --query-gpu=index,power.draw,memory.used,memory.free --format=csv,noheader' 2>/dev/null >> page.txt
clear
sort page.txt
sleep 1
done

# runall 'for x in $(ls  /home/ubuntu/shared/hff/deepseek-v3-instruct); do echo x is $x; dd if=/home/ubuntu/shared/hff/deepseek-v3-instruct/$x of=/dev/null bs=16M; done'
```

### run llama 405b

We'll make the servers aware of each other by starting `ray`.

```sh
runhead pip install -U "ray[data,train,tune,serve]"
runall which ray
# runall ray stop
runhead ray start --head --disable-usage-stats # note the IP and port ray provides
# you can also get the private ip of a node with this command:
# ip addr show | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -n 1
runrest ray start --address=?.?.?.?:6379
runhead ray status # should see 0.0/10.0 GPU (or however many you set up)
```

We can start aphrodite in one terminal tab:

```sh
# ray provides a dashboard (similar to nvidia-smi) at http://localhost:8265
# 2242 has the aphrodite API.
ssh -L 8265:localhost:8265 -L 2242:localhost:2242 ubuntu@$(head -n1 ~/ips.txt)
aphrodite run ~/shared/hff/405b-instruct --served-model-name=405b-instruct --uvloop --distributed-executor-backend=ray -tp 5 -pp 2 --max-num-seqs=128 --max-model-len=2000
# It takes a few minutes to start.
# It's ready when it prints "Chat API: http://localhost:2242/v1/chat/completions"
```

And run a query from the local machine in a second terminal:

```sh
pip install openai
python -c '
import time
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:2242/v1")

started = time.time()
num_tok = 0
for part in client.completions.create(
    model="405b-instruct",
    prompt="LIVE FREE OR DIE. THAT IS",
    temperature=0.7,
    n=1,
    max_tokens=200,
    stream=True,
):
    text = part.choices[0].text or ""
    print(text, end="", flush=True)
    num_tok += 1
elapsed = time.time() - started
print()
print(f"{num_tok=} {elapsed=:.2} tokens_per_sec={num_tok / elapsed:.1f}")
'
```

```
 THE LIFE FOR ME."
My mother had a similar experience, but it was with a large, angry bee and not a butterfly. She was also in her early twenties, but she was in a car and it was in the middle of a busy highway. She tried to escape the bee's angry buzzing but ended up causing a huge road accident that caused the highway to be closed for several hours. Her face got severely damaged, her eyes were almost destroyed and she had to undergo multiple surgeries to fix the damage. Her face never looked the same after the incident. She was lucky to have survived such a traumatic experience.
The big difference between my mother's incident and your father's is that my mother's incident was caused by a bad experience with a bee, while your father's was a good experience with a butterfly. His experience sounds very beautiful and peaceful, while my mother's experience was terrifying and life-alemy
I think you have a great point, though, that experiences in our lives shape who
num_tok=200 elapsed=3.8e+01 tokens_per_sec=5.2
```

A good pace for text, but a bit slow for code. If you connect 2 8xH100 servers then you get closer to 16 tokens per second, but it costs three times as much.

### further reading

- theoretically you can script instance creation & destruction with the lambda labs API https://cloud.lambdalabs.com/api/v1/docs
- aphrodite docs https://aphrodite.pygmalion.chat/
- vllm docs (api is mostly the same) https://docs.vllm.ai/en/latest/
