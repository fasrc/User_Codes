FROM ubuntu:18.04
RUN apt update && apt -y upgrade
RUN apt -y install sudo wget
RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && sudo dpkg -i packages-microsoft-prod.deb
RUN sudo apt-get install -y apt-transport-https && sudo apt-get update && sudo apt-get install -y dotnet-sdk-2.1
RUN sudo apt-get install libexpat1 libjpeg62 libpng16-16 libxml2
