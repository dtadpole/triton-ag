1. Install OpenSSH client (if not already)
sudo dnf install -y openssh-clients

2. Create a systemd user service for ssh-agent
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/ssh-agent.service

--- BEGIN OF FILE ---
[Unit]
Description=SSH key agent

[Service]
Type=simple
Environment=SSH_AUTH_SOCK=%t/ssh-agent.socket
ExecStart=/usr/bin/ssh-agent -D -a $SSH_AUTH_SOCK

[Install]
WantedBy=default.target
--- END OF FILE ---

3. Enable & start the service
systemctl --user daemon-reload
systemctl --user enable ssh-agent
systemctl --user start ssh-agent

4. Export environment for your shell
export SSH_AUTH_SOCK="$XDG_RUNTIME_DIR/ssh-agent.socket"

source ~/.bashrc

5. Add your keys
ssh-add ~/.ssh/id_rsa

ssh-add -l


