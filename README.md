# Serving deepseek r1 & v3 on GH200s

Lambda labs still has half-off GH200s as a promotion to get more people used to the ARM tooling. I [previously](https://dev.to/qpwo/how-to-run-llama-405b-bf16-with-gh200s-7da) wrote a tutorial for running llama 405b, but deepseek r1 is clearly a better model, so let's serve it instead. There's a couple differences from serving llama:

- You need 12 or 16 GPUs to have decent throughput. You can still use 8 if you don't need much throughput.
- VLLM works better than aphrodite for deepseek right now. In fact they just released an [update](https://github.com/vllm-project/vllm/releases/tag/v0.7.2) that makes deepseek inference quite a bit faster. Roughly a 40% throughput improvement.
  - Normally a 16x 1x gpu cluster would be pretty slow for inference but for some reason this actually works pretty well.

## Create instances

This time, instead of using the website, let's use the API to make the instances.

Make a [filesystem](https://cloud.lambdalabs.com/file-systems) called 'shared' in Washington, DC and write down the name of your [ssh key](https://cloud.lambdalabs.com/ssh-keys). Also generate an [api key](https://cloud.lambdalabs.com/api-keys) and save it.

```sh
export sshkey_name=my-key
export shared_fs_name=shared
export LAMBDA_API_KEY="..."
```

This script will make 16 GH200 instances. Ignore the rate limit errors, it will keep retrying until 16 have been made.

```sh
function lambda-api {
    local method=$1
    local route=$2
    shift 2
    curl --fail -q -X "$method" -H "Authorization: Bearer $LAMBDA_API_KEY" "https://cloud.lambdalabs.com/api/v1/$route" -H "Content-Type: application/json" "$@"
}
export -f lambda-api

num_want=16
num_got=0
while [[ $num_got -lt $num_want ]]; do
    lambda-api POST instance-operations/launch -d '{
        "region_name": "us-east-3",
        "instance_type_name": "gpu_1x_gh200",
        "sshkey_names": ["'$sshkey_name'"],
        "file_system_names": ["'$shared_fs_name'"],
        "quantity": 1,
        "name": "node_'$num_got'"
    }' && ((num_got++))
    echo num_got=$num_got
    sleep 3
done
```

After all the instances are created, copy all the IP addresses from the [instances page](https://cloud.lambdalabs.com/instances) and save it to `~/ips.txt`.

![copy-ips](shot1.png)

## Bulk ssh connection helpers

I prefer direct bash & ssh over anything fancy like kubernetes or slurm. It's manageable with some helpers.

```sh
# save all the ssh fingerprints now to skip confirmation later
for ip in $(cat ~/ips.txt); do
    echo "doing $ip"
    ssh-keyscan $ip >> ~/.ssh/known_hosts
done

export runprefix=""
function runip() {
    ssh -i ~/.ssh/lambda_id_ed25519 ubuntu@$ip -- stdbuf -oL -eL bash -l -c "$(printf "%q" "$runprefix""$*")" < /dev/null
}
function runk() { ip=$(sed -n "$((k + 1))"p ~/ips.txt) runip "$@"; }
function runhead() { ip="$(head -n1 ~/ips.txt)" runip "$@"; }
function runips() {
    local pids=()
    for ip in $ips; do
        ip=$ip runip "$@" |& sed "s/^/$ip\t /" &
        pids+=($!)
    done
    wait "${pids[@]}" &>/dev/null
}
function runall() { ips="$(cat ~/ips.txt)" runips "$@"; }
function runrest() { ips="$(tail -n+2 ~/ips.txt)" runips "$@"; }

function sshk() {
    ip=$(sed -n "$((k + 1))"p ~/ips.txt)
    ssh -i ~/.ssh/lambda_id_ed25519 ubuntu@$ip
}
alias ssh_head='k=0 sshk'

function killall() {
    pkill -ife 192.222
}
```

Let's check that it works

```sh
runall echo ok
```

## Set up NFS cache

We'll be putting the python environment in the NFS. It will load much faster if we cache it.

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

## Install python 3.11

I've had better luck with `apt` packages than `conda` ones, so I'll use `apt` in this tutorial.

```sh
runall sudo add-apt-repository ppa:deadsnakes/ppa -y
runall sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils
runall python3.11 --version
```

## Create virtualenv

Instead of carefully doing the exact same commands on every machine, we can use a virtual environment in the NFS and just control it with the head node. This makes it a lot easier to correct mistakes.

```sh
runhead python3.11 -m venv --copies shared/myvenv
runhead 'source shared/myvenv/bin/activate; which python'
export runprefix='source shared/myvenv/bin/activate ; ' # this is used by runip()
runall which python
```

## Download models

I hit some kind of networking issue when I used more workers, but you can download r1 and v3 in about 20 minutes each this way. You can let this run in the background while you do the next step in a separate terminal. I think you get faster downloads if you use a [huggingface token](https://huggingface.co/settings/tokens).

```sh
runhead "pip install hf_transfer 'huggingface_hub[hf_transfer]'"
runall "huggingface-cli login --token ..."
runall "export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download --max-workers=1 --local-dir ~/dsr1 deepseek-ai/DeepSeek-R1
huggingface-cli download --max-workers=1 --local-dir ~/dsv3 deepseek-ai/DeepSeek-V3"
```

I spent a few hours writing a script to have each server download a part of the model, then synchronize it between all servers, but it wasn't actually much faster. Just download straight from hf.

## Install VLLM

### From my wheels

```sh

```

### From source

```sh
runhead pip install ninja setuptools build setuptools_scm wheel bindings build cmake
runhead "pip install 'numpy<2' torch==2.5.1 --index-url 'https://download.pytorch.org/whl/cu124'"

runhead mkdir ~/git
runhead git clone https://github.com/vllm-project/vllm ~/git/vllm
runhead 'cd ~/git/vllm && git checkout v0.7.2'
runhead 'cd ~/git/vllm && python -m build --no-isolation --wheel --verbose .'
# this will take about 20 minutes
```

Don't waste your time trying to configure parallel builds, ninja defaults are basically optimal.

## Serve it!

The nodes communicate using [ray](https://docs.ray.io/en/latest/index.html), so first we'll start a ray cluster.

```sh
runhead ray start --head
```

This will give you the private/LAN IP address you use to connect the other nodes.

```sh
runrest ray start --address=...
```

And now we can fire up vllm. The first run will cause weights to be cached in RAM, so the second time you start vllm will be faster.

```sh
runhead vllm serve
```

## Load testing
