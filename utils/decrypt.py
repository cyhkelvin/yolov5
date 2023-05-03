# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Download utils
"""

from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
from io import BytesIO
from pathlib import Path

import torch


# if byte_string less than 16n digits, padding to 16
def add_to_16(byte_string):
    if len(byte_string) % 16:
        add = 16 - (len(byte_string) % 16)
    else:
        add = 0
    byte_string = byte_string + (b'\0' * add)
    return byte_string


# 加密函数
def Encrypt(byte_string, key='0123456789abcdef'.encode('utf-8'), iv=b'qqqqqqqqqqqqqqqq', mode=AES.MODE_CBC):
    byte_string = add_to_16(byte_string)
    cryptos = AES.new(key, mode, iv)
    cipher_text = cryptos.encrypt(byte_string)
    # 因为AES加密后的字符串不一定是ascii字符集的，输出保存可能存在问题，所以这里转为16进制字符串
    return b2a_hex(cipher_text)


# 解密后，去掉补足的空格用strip() 去掉
def Decrypt(byte_string, key='0123456789abcdef'.encode('utf-8'), iv=b'qqqqqqqqqqqqqqqq', mode=AES.MODE_CBC):
    cryptos = AES.new(key, mode, iv)
    plain_text = cryptos.decrypt(a2b_hex(byte_string))
    return plain_text.rstrip(b'\0')


def attempt_decrypt(file, key='0123456789abcdef'.encode('utf-8'), iv=b'qqqqqqqqqqqqqqqq'):
    # Attempt file download from GitHub release assets if not found locally. release = 'latest', 'v7.0', etc.
    from utils.general import LOGGER

    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        LOGGER.error(f'file {file} not Found!')
    else:
        byte_string = open(str(file), 'rb').read()
        decrypted = BytesIO(Decrypt(byte_string, key))
        return decrypted
