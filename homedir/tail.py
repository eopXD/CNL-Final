import subprocess
import codecs

def tailPacket ( filename: str, lines: int ) :
    return (codecs.decode(subprocess.check_output(["tail", "-n "+str(lines), filename]), encoding='utf-8').split())

# packet format:
# [timestamp(nanosecond), timestamp(microsecond), packet header, 強度, 頻率 (不理), 長度, sha256]
def getPacket ( filename: str ) -> list :
    raw_data = tailPacket(filename, 100)
    data = [ line.split(",") for line in raw_data ]
    return (data) # array of packets

a = getPacket("packets1")
b = getPacket("packets2")
