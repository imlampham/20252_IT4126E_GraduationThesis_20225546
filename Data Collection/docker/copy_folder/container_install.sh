#!/usr/bin/bash

# 1. Cập nhật hệ thống và cài các công cụ lõi
apt-get update
apt-get install -y software-properties-common python3-pip wget tcpdump

# 2. Cài đặt tshark 
# (BẮT BUỘC GIỮ LẠI: Vì thư viện 'pyshark' của Python cần tshark cài sẵn trên Linux để đọc ruột file .pcap)
add-apt-repository -y ppa:wireshark-dev/stable
DEBIAN_FRONTEND=noninteractive apt-get install -y tshark

# 3. Cài đặt Google Chrome (Cho trình duyệt Headless)
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
DEBIAN_FRONTEND=noninteractive apt-get install -y ./google-chrome-stable_current_amd64.deb
rm ./google-chrome-stable_current_amd64.deb # Xóa file rác sau khi cài xong

# 4. Cài đặt các thư viện Python THỰC SỰ cần thiết
pip3 install pyshark selenium nest_asyncio