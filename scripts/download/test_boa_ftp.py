from ftplib import FTP

HOST = "data.argo.org.cn"

ftp = FTP(HOST, timeout=60)

print("Connecting...")

ftp.login("ftp", "aadityasmitha@gmail.com")

print("Connected!")

ftp.cwd("/pub/ARGO/BOA_Argo/")

print("\nContents:\n")

files = ftp.nlst()

for f in files:
    print(f)

ftp.quit()