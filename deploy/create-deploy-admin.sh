#!/usr/bin/env bash
set -euo pipefail

user="deploy-admin"
key_stage="/tmp/deploy-admin-authorized-key.pub"

if ! id "$user" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash --groups sudo "$user"
fi

install -d -m 700 -o "$user" -g "$user" "/home/$user/.ssh"
install -m 600 -o "$user" -g "$user" "$key_stage" "/home/$user/.ssh/authorized_keys"
passwd --lock "$user" >/dev/null

printf '%s\n' "$user ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/$user"
chmod 440 "/etc/sudoers.d/$user"
visudo -cf "/etc/sudoers.d/$user"

rm -f "$key_stage"
printf 'ACCOUNT='
id -un "$user"
printf 'HOME_MODE='
stat -c '%a' "/home/$user"
printf 'SSH_MODE='
stat -c '%a' "/home/$user/.ssh"
printf 'AUTH_KEYS_MODE='
stat -c '%a' "/home/$user/.ssh/authorized_keys"
