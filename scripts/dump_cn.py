import xml.etree.ElementTree as ET
import os

here = os.path.dirname(os.path.abspath(__file__))
root = ET.parse(r"D:\Artensoft Photo Collage Maker Pro11\Chinese.xml").getroot()
out = os.path.join(here, "..", "_ref", "chinese_strings.txt")
n = 0
with open(out, "w", encoding="utf-8") as f:
    for el in root.find("Strings"):
        text = (el.text or "").strip()
        f.write(el.tag + " = " + text + "\n")
        n += 1
print("dumped", n, "strings to", out)
