"""Official standard administrative boundaries: 自然资源部标准地图服务 审图号 GS（2024）0650号.
Geometry used as-is (EPSG:4326, UTF-8); no simplification. Only display-level dissolve for the national outline."""
from pathlib import Path
import geopandas as gpd
D = Path("/workspace/apple_orchard_project/data/official_boundary/GS2024_0650")
GS_NO = "GS（2024）0650号"
CREDIT = "底图采用自然资源部标准地图服务系统下载的标准地图（审图号：GS（2024）0650号）制作，底图边界无修改"

def prov():   return gpd.read_file(D / "china_prov.shp")
def pref():   return gpd.read_file(D / "china_pref.shp")
def county(): return gpd.read_file(D / "china_county.shp")
def prov_lines(): return gpd.read_file(D / "china_provline.shp")   # 境界线: 南海断续线、未定国界等

ZHOUZHI_GB = [156654002, 156654003, 156654004, 156654021, 156654022, 156654023, 156654024,
              156654025, 156654026, 156654027, 156654028]      # 伊犁州直 11 县市
KEKEDALA_GB = 156659008                                          # 可克达拉市（兵团第四师，位于河谷内）

def ili_zhouzhi():
    c = county(); c["gb"] = c.gb.astype("int64"); return c[c.gb.isin(ZHOUZHI_GB)].copy()
def kekedala():
    c = county(); c["gb"] = c.gb.astype("int64"); return c[c.gb == KEKEDALA_GB].copy()
def gongliu():
    c = county(); return c[c.name == "巩留县"].copy()
def xinjiang():
    p = prov(); return p[p.name.str.contains("新疆")].copy()
