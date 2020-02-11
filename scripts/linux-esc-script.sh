#!/bin/bash

# run this to make wsl-executable file endings:
# tr -d "\015" <esc-script.sh > linux-esc-script.sh       

# run :
# ./linux-esc-script.sh <file.rawlog> | less       

# or, run:
# ./linux-esc-script.sh <file.rawlog>  >  <file.log>

# echo 'hey'

sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?m//g"  $1 | \
sed -r "s/\x1B\[.*G//g" | \
sed 's/[[:cntrl:]]//g' 