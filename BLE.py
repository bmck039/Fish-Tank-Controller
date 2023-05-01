import sys
from bleak import BleakScanner
from bleak import BleakClient


# def hexToBin(hex):

def ble_encode(b):
  raw_len = len(b)
  rand = 0
  encoded_bytes = bytearray([0x54, (raw_len + 1) ^ 0x54, rand ^ 0x54])
  for byte in b:
    encoded_bytes.append(byte ^ rand)

  return encoded_bytes

def ble_decode(b):
  iv = b[0]
  length = b[1] ^ iv
  key = b[2] ^ iv

  decoded_bytes = bytearray()
  for i in range(3, len(b)):
    decoded_bytes.append(b[i] ^ key)

  return decoded_bytes

# The last byte of a message is a CRC value that is just every byte XOR'd in order.
def crc(cmd):
  check = 0
  for i in range(0, len(cmd)):
    check = check ^ cmd[i]
  return check

def buildMessage(raw_bytes):
  raw_msg = bytearray(raw_bytes)
  # Prepend message header (0x68), aka FRM_HDR in apk source code
  msg = bytearray([0x68])
  msg.extend(raw_msg)
  msg.append(crc(msg))
  print("Dec message: ", msg.hex())
  enc_msg = ble_encode(msg)
  print("Enc message: ", enc_msg.hex())
  return enc_msg

def getPowerOnMessage():
  # Power:
  #  CMD_SWITCH (0x03), [0|1]
  return buildMessage([0x03, 0x01])


def getPowerOffMessage():
  # Power:
  #  CMD_SWITCH (0x03), [0|1]
  return buildMessage([0x03, 0x00])

def getModeManualMessage():
    return buildMessage([0x02, 0x00])

def getModeAutoMessage():
    return buildMessage([0x02, 0x01])

def getModeProMessage():
    return buildMessage([0x02, 0x02])

def setLightPro(list):
  object0 = list[0]
  print(list)
  length = len(object0['color'])
  i = length + 2
  size = (len(list) * i) + 4
  bArr = bytearray(size)
  bArr[0] = 104 #redundant, this value is added in buildMessage
  bArr[1] = 16
  bArr[2] = len(list)
  for i2 in range(len(list)):
    i3 = i2 * i
    lightObject = list[i2]
    bArr[i3 + 3] = int(lightObject['time'] // 60)
    bArr[i3 + 4] = round(lightObject['time'] % 60)
    for i4 in range(length):
      bArr[i3 + 5 + i4] = lightObject['color'][i4]
  print(bArr)
  del bArr[0] #first item is redundant and added in buildMessage and adding it twice is bad
  return buildMessage(bArr)

# Sets the brightness of one or more channels
# Level: 0-1000 (0x03E8) -- note this is two bytes and is big-endian
# Channels not specified will not be modified.
def getBrightnessMessage(red=False, blue=False, cwhite=False, pwhite=False, wwhite=False):
  # Channel brightness message format:
  #   CMD_CTRL (0x04), <16-bit red>, <16-bit blue>, <16-bit cwhite>, <16-bit pwhite>, <16-bit wwhite>
  # Notes: Values set to 0xFFFF will not modify anything.
  #        Legal range is 0x0000-0x03E8, big-endian.

  def consider(color):
    nop = b'\xff\xff'
    if color is False:
      return nop
    elif color < 0 or color > 1000:
      print("fatal: brightness values must be between 0-1000")
      sys.exit(1)
    else:
      return color.to_bytes(2, byteorder='big')

  cmd = bytearray([0x04])
  cmd.extend(consider(red))
  cmd.extend(consider(blue))
  cmd.extend(consider(cwhite))
  cmd.extend(consider(pwhite))
  cmd.extend(consider(wwhite))

  return buildMessage(cmd)

async def sendSchedule(schedule):
    print("discovering")
    devices = await BleakScanner.discover()
    for d in devices:
        if d.name == "Plant 3.0":
            address = d.address
            BLEDevice = BleakClient(address)
            try:
                await BLEDevice.connect()
                mode = getModeProMessage()
                await BLEDevice.write_gatt_char("00001001-0000-1000-8000-00805f9b34fb", mode)
                data = setLightPro(schedule)
                print(data)
                await BLEDevice.write_gatt_char("00001001-0000-1000-8000-00805f9b34fb", data)

            finally:
                await BLEDevice.disconnect()