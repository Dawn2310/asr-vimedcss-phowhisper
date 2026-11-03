"""Post-process: chart blanks, one distinct transition per slide."""
import re, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
TR = {  # slide number -> transition element (slide 1 has none)
    2: '<p:fade/>', 3: '<p:push dir="l"/>', 4: '<p:wipe dir="r"/>', 5: '<p:split orient="vert" dir="out"/>', 6: '<p:blinds dir="vert"/>',
    7: '<p:cover dir="l"/>', 8: '<p:pull dir="u"/>', 9: '<p:circle/>', 10: '<p:diamond/>', 11: '<p:wheel spokes="4"/>', 12: '<p:zoom dir="in"/>',
    13: '<p:comb dir="horz"/>', 14: '<p:strips dir="rd"/>', 15: '<p:wedge/>', 16: '<p:plus/>', 17: '<p:checker dir="horz"/>', 18: '<p:randomBar dir="vert"/>',
    19: '<p:dissolve/>', 20: '<p:push dir="u"/>', 21: '<p:wipe dir="d"/>', 22: '<p:cover dir="r"/>', 23: '<p:split orient="horz" dir="in"/>', 24: '<p:fade thruBlk="1"/>',
}
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
for it in zin.infolist():
    data = zin.read(it.filename)
    n = it.filename
    m = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", n)
    if m and int(m.group(1)) in TR:
        x = data.decode("utf-8")
        assert "</p:clrMapOvr>" in x, n
        x = x.replace("</p:clrMapOvr>", '</p:clrMapOvr><p:transition spd="med">%s</p:transition>' % TR[int(m.group(1))], 1)
        data = x.encode("utf-8")
    elif re.fullmatch(r"ppt/charts/chart\d+\.xml", n):
        x = data.decode("utf-8")
        x, k = re.subn(r'<c:pt idx="\d+"><c:v></c:v></c:pt>', "", x)
        x = re.sub(r'<c:dispBlanksAs val="\w+"/>', '<c:dispBlanksAs val="span"/>', x)
        print(n, "removed empty points:", k, "span" , 'dispBlanksAs val="span"' in x)
        data = x.encode("utf-8")
    zout.writestr(it, data)
zout.close()
print("ok", dst)
