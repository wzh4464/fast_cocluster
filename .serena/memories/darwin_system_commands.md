# Darwin (macOS) System Commands

## File System Operations

### Navigation
```bash
# Change directory
cd /path/to/directory

# Go to home directory
cd ~

# Go back one directory
cd ..

# Print working directory
pwd
```

### Listing Files
```bash
# List files
ls

# List with details
ls -l

# List all (including hidden)
ls -a

# List all with details
ls -la

# List sorted by modification time
ls -lt

# Human-readable sizes
ls -lh
```

### File Operations
```bash
# Copy file
cp source.txt destination.txt

# Copy directory recursively
cp -r source_dir/ dest_dir/

# Move/rename file
mv old_name.txt new_name.txt

# Remove file
rm file.txt

# Remove directory recursively
rm -rf directory/

# Create directory
mkdir new_directory

# Create nested directories
mkdir -p path/to/nested/directory
```

### File Viewing
```bash
# View file contents
cat file.txt

# View with line numbers
cat -n file.txt

# View large files (paginated)
less file.txt

# View first 10 lines
head file.txt

# View last 10 lines
tail file.txt

# Follow file updates (useful for logs)
tail -f logfile.txt
```

## Text Processing

### Search
```bash
# Search in file
grep "pattern" file.txt

# Search recursively in directory
grep -r "pattern" directory/

# Case-insensitive search
grep -i "pattern" file.txt

# Show line numbers
grep -n "pattern" file.txt

# Search for whole word
grep -w "word" file.txt

# Count matches
grep -c "pattern" file.txt
```

### Find Files
```bash
# Find by name
find . -name "*.rs"

# Find by type (file)
find . -type f -name "*.rs"

# Find by type (directory)
find . -type d -name "target"

# Find and execute command
find . -name "*.rs" -exec wc -l {} +

# Case-insensitive find
find . -iname "readme.md"
```

### Text Manipulation
```bash
# Replace text (BSD sed - macOS)
sed 's/old/new/g' file.txt

# Replace in-place (requires empty string for backup)
sed -i '' 's/old/new/g' file.txt

# Extract columns (by space)
awk '{print $1, $3}' file.txt

# Count lines, words, characters
wc file.txt

# Count lines only
wc -l file.txt

# Sort lines
sort file.txt

# Unique lines
sort file.txt | uniq
```

## Process Management

### Viewing Processes
```bash
# List all processes
ps aux

# Filter processes
ps aux | grep rust

# Interactive process viewer
top

# Better process viewer (if installed)
htop

# Show process tree
pstree
```

### Managing Processes
```bash
# Run in background
cargo build &

# Bring to foreground
fg

# List background jobs
jobs

# Kill process by PID
kill <PID>

# Force kill
kill -9 <PID>

# Kill by name
killall cargo

# Suspend process
kill -STOP <PID>

# Resume process
kill -CONT <PID>
```

## System Information

### System Stats
```bash
# System information
uname -a

# OS version
sw_vers

# CPU information
sysctl -n machdep.cpu.brand_string

# Memory information
vm_stat

# Disk usage
df -h

# Directory size
du -sh directory/

# Free memory (macOS specific)
top -l 1 | grep PhysMem
```

### Networking
```bash
# IP address
ifconfig

# Test connectivity
ping google.com

# DNS lookup
nslookup google.com

# Network connections
netstat -an

# Active internet connections
lsof -i

# Port usage
lsof -i :8080
```

## Compression & Archives

### Tar Archives
```bash
# Create tar.gz
tar -czf archive.tar.gz directory/

# Extract tar.gz
tar -xzf archive.tar.gz

# List contents
tar -tzf archive.tar.gz

# Create tar (no compression)
tar -cf archive.tar directory/

# Extract tar
tar -xf archive.tar
```

### Zip Archives
```bash
# Create zip
zip -r archive.zip directory/

# Extract zip
unzip archive.zip

# List contents
unzip -l archive.zip
```

## Package Management (macOS)

### Homebrew
```bash
# Install package
brew install <package>

# Update Homebrew
brew update

# Upgrade packages
brew upgrade

# Search for package
brew search <package>

# List installed packages
brew list

# Remove package
brew uninstall <package>

# Clean up old versions
brew cleanup
```

## Permissions

### File Permissions
```bash
# Make executable
chmod +x script.sh

# Change permissions (numeric)
chmod 755 file.sh

# Change owner
chown user:group file.txt

# Change recursively
chmod -R 755 directory/
chown -R user:group directory/
```

## Environment & Shell

### Environment Variables
```bash
# Print environment variables
env

# Print specific variable
echo $PATH

# Set variable (current session)
export MY_VAR="value"

# Add to PATH
export PATH=$PATH:/new/path
```

### Shell Configuration
```bash
# Reload shell config (zsh - default on macOS)
source ~/.zshrc

# Edit zsh config
vim ~/.zshrc
# or
nano ~/.zshrc

# Edit bash config (if using bash)
vim ~/.bashrc
```

## Rust-Specific (macOS)

### Rustup
```bash
# Update Rust toolchain
rustup update

# Switch to nightly
rustup default nightly

# Switch to stable
rustup default stable

# Show installed toolchains
rustup show

# Add component
rustup component add clippy rustfmt
```

### Cargo
```bash
# Check cargo home
echo $CARGO_HOME

# Clear cargo cache
cargo clean

# Update cargo itself
rustup update
```

## Useful Aliases for ~/.zshrc

```bash
# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ll='ls -la'
alias l='ls -l'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline --graph'

# Cargo shortcuts
alias cb='cargo build'
alias ct='cargo test'
alias cr='cargo run'
alias cf='cargo fmt'
alias cc='cargo clippy'

# System
alias ports='lsof -i -P | grep LISTEN'
alias cleanup='brew cleanup && cargo clean'
```

## Darwin-Specific Notes

1. **BSD vs GNU tools**: macOS uses BSD versions of common tools (sed, awk, grep), which have slightly different syntax than GNU versions on Linux

2. **Case sensitivity**: macOS file system is case-insensitive by default (but can be case-sensitive on APFS)

3. **Xcode Command Line Tools**: Required for many development tasks
   ```bash
   xcode-select --install
   ```

4. **Open command**: Open files/directories in default application
   ```bash
   open file.txt
   open .  # Open current directory in Finder
   open -a "Visual Studio Code" .  # Open in specific app
   ```

5. **pbcopy/pbpaste**: Clipboard utilities
   ```bash
   cat file.txt | pbcopy  # Copy to clipboard
   pbpaste > file.txt     # Paste from clipboard
   ```