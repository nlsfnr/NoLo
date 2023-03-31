#!/bin/bash

# This script sets up rsync on a remote server with a specified username and IP
# address, creating a directory to hold project files, copying the files to the
# remote server using rsync and setting ownership on the directory. This script
# assumes that rsync and ssh are already installed on the local machine and the
# remote server.
#
# Usage: ./setup_rsync.sh username ip_address
#
# Replace "username" with your username on the remote server, and "ip_address"
# with the IP address of the remote server.

# Check if the user has provided a username and IP address
if [ $# -lt 2 ]; then
    echo "Usage: $0 username ip_address"
    exit 1
fi

# Read the username and IP address from the command-line arguments
USERNAME=$1
IP_ADDRESS=$2

# Create a directory to hold the project files
echo "Creating directory on remote server..."
ssh $USERNAME@$IP_ADDRESS "mkdir -p ~/NoLo"

# Copy the project files to the remote server
echo "Copying files to remote server..."
rsync -avz --exclude-from=.gitignore . $USERNAME@$IP_ADDRESS:~/NoLo/

# Set ownership and permissions on the project directory
echo "Setting ownership on remote server..."
ssh $USERNAME@$IP_ADDRESS "sudo chown -R $USERNAME:$USERNAME ~/NoLo"

echo "Done!"
