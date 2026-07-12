Ansible playbook for deploying BitNet service

Usage:

- Set up an inventory with group `bitnet_hosts` pointing to target hosts.
- Copy repository to `/opt/bitnet` on target hosts, or adjust `repo_path` var.
- Run:

  ansible-playbook -i inventory deploy/ansible/bitnet.yml -e "resource_profile=medium"

Resource profiles:
- small: for low-RAM/CPU machines
- medium: default balance
- large: for powerful servers
