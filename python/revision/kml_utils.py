# -*- coding: utf-8 -*-
"""Minimal KML writer (polygons + labelled points in coloured folders) for Google Earth checking kits."""
from xml.sax.saxutils import escape
def _ring(coords): return " ".join(f"{x:.7f},{y:.7f},0" for x, y in coords)
def _poly(g):
    parts = [g] if g.geom_type == "Polygon" else list(g.geoms)
    out = []
    for p in parts:
        inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{_ring(r.coords)}</coordinates></LinearRing></innerBoundaryIs>" for r in p.interiors)
        out.append(f"<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{_ring(p.exterior.coords)}</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")
    return out[0] if len(out) == 1 else "<MultiGeometry>" + "".join(out) + "</MultiGeometry>"
def style(sid, line_abgr, fill_abgr="00ffffff", width=2.5, label_scale=0.9):
    return (f'<Style id="{sid}"><LineStyle><color>{line_abgr}</color><width>{width}</width></LineStyle><PolyStyle><color>{fill_abgr}</color></PolyStyle>'
            f'<LabelStyle><color>{line_abgr}</color><scale>{label_scale}</scale></LabelStyle><IconStyle><color>{line_abgr}</color><scale>0.7</scale>'
            f'<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>')
def write_kml(path, name, styles, folders, description=""):
    """folders: list of (folder_name, [ {name, desc, geom (shapely polygon|None), point (x,y)|None, style} ])"""
    s = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>', f"<name>{escape(name)}</name>",
         f"<description>{escape(description)}</description>"] + styles
    for fname, items in folders:
        s.append(f"<Folder><name>{escape(fname)}</name><open>1</open>")
        for it in items:
            geo = []
            if it.get("geom") is not None: geo.append(_poly(it["geom"]))
            if it.get("point") is not None: geo.append(f"<Point><coordinates>{it['point'][0]:.7f},{it['point'][1]:.7f},0</coordinates></Point>")
            g = geo[0] if len(geo) == 1 else "<MultiGeometry>" + "".join(geo) + "</MultiGeometry>"
            s.append(f"<Placemark><name>{escape(it['name'])}</name><description>{escape(it.get('desc', ''))}</description><styleUrl>#{it['style']}</styleUrl>{g}</Placemark>")
        s.append("</Folder>")
    s.append("</Document></kml>"); open(path, "w", encoding="utf-8").write("\n".join(s))
