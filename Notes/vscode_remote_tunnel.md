# Set Up VSCode Remote Tunnel For Cannon

## Using CLI

1. Login to Cannon

2. Execute the following two commands:

```bash
[jharvard@boslogin02 ~]$ curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64\
		     --output vscode_cli.tar.gz
[jharvard@boslogin02 ~]$ tar -xf vscode_cli.tar.gz

```

3. An executable, `code`, will be generated in your current working
directory. Either keep it in that or your $HOME or move it to your
LABS folder, e.g.,

  `mv code /n/holylabs/LABS/<PI_Lab>/<Desired-Folder>/`

4. Add the path to your `~/.bashrc` so that the executable is always
  available to you regardless of the node you are on. Add the
  following line to your `~/.bashrc`

  `export PATH=/n/holylabs/LABS/<PI_Lab>/<Desired-Folder>:$PATH`

5. Save `~/.bashrc`, and execute: `source ~/.bashrc` on the terminal
  prompt

6. Go to a compute node, e.g.,:
  `[jharvard@boslogin02 ~]$ salloc -p gpu_test --gpus 1 --mem 10000 -t 0-01:00`

7. Execute the following command:
  `[jharvard@boslogin02 ~]$ code tunnel`

8. Follow the screen message and log in using either your Github or
  Microsoft account, e.g.: Github Account

9. To grant access to the server, open the URL
  `https://github.com/login/device` and copy-paste the code given on
  the screen

10. Name the machine, say, `cannoncompute`

11. Open the link that appears in your local browser and follow the
  authentication process as mentioned in steps# 3& 4 of [using the
  code
  cli](https://code.visualstudio.com/docs/remote/tunnels#_using-the-code-cli)

12. On your local VSCode, install Remote Tunnel extension 

13. Click on VS Code Account menu, choose **Turn on Remote Tunnel Access**

14. Click on `cannoncompute` to get connected to the remote machine. Prior
  to clicking, make sure you see:

  `Remote -> Tunnels -> cannoncompute running`

15. Enjoy your work using your local VSCode on a Cannon's compute node


==**_Note:_** Every time you access a compute node, the executable,
	`code`, will be in your path. However, you will have to repeat
	step#9 before executing step#14 above.==


## Resource(s)
* [VSCode Remote Tunnel](https://code.visualstudio.com/docs/remote/tunnels)
