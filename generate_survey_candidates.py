from pathlib import Path
import json, html, math
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform as shp_transform
from scipy.spatial import cKDTree

BASE = Path('/workspace/apple_orchard_project')
OUT = BASE/'outputs/revision/field_campaign'
OUT.mkdir(parents=True, exist_ok=True)
R19 = BASE/'data/aomc/AOMC_30_Xinjiang_2019.tif'
R20 = BASE/'data/aomc/AOMC_30_Xinjiang_2020.tif'
R21 = BASE/'data/aomc/AOMC_30_Xinjiang_2021.tif'
ROI = BASE/'data/roi/Yili_county_full.geojson'
EXCL = BASE/'data/field_survey/usable_candidates/ili_field_apple_usable.csv'
KIT = BASE/'data/interpretation_kit/Gongliu_independent_val_points.geojson'
SEED = 20260925
MIN_SEP = 450.0
EXCL_DIST = 300.0
TARGET = 175
NOTE = 'AOMC候选点，仅用于野外核查，不代表地面真值；请现场确认是否为苹果园。'

# Existing usable field apple points.
df = pd.read_csv(EXCL)
loncol = '_lon' if '_lon' in df.columns else '经度'
latcol = '_lat' if '_lat' in df.columns else '纬度'
excl_ll = df[[loncol,latcol]].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(float)
# Conservatively avoid Gongliu interpretation-kit points flagged as AOMC-apple hints.
kit = json.loads(KIT.read_text())
kit_ll = []
for feat in kit.get('features', []):
    p = feat.get('properties', {})
    if str(p.get('aomc2021_hint', '')) in {'1','1.0','True','true'}:
        co = feat.get('geometry', {}).get('coordinates', [])
        if len(co) >= 2: kit_ll.append([float(co[0]), float(co[1])])
if kit_ll:
    excl_ll = np.vstack([excl_ll, np.asarray(kit_ll, float)])

counties = json.loads(ROI.read_text())['features']
# AOMC CRS is a projected equal-area CRS; use it for metric distances.
with rasterio.open(R21) as s21, rasterio.open(R19) as s19, rasterio.open(R20) as s20:
    tf = Transformer.from_crs('EPSG:4326', s21.crs, always_xy=True)
    exx, exy = tf.transform(excl_ll[:,0], excl_ll[:,1])
    excl_xy = np.column_stack([exx,exy])
    excl_tree = cKDTree(excl_xy) if len(excl_xy) else None
    rng = np.random.default_rng(SEED)
    bycounty = {}
    summary = {}
    for feat in counties:
        county = feat['properties']['name']
        geom = shp_transform(tf.transform, shape(feat['geometry']))
        arr, tr = mask(s21, [geom], crop=True, filled=False)
        ma = arr[0]
        a = np.asarray(ma)
        valid = (~np.ma.getmaskarray(ma)) & (a > 0)
        ys, xs = np.where(valid)
        if len(xs)==0:
            bycounty[county] = {'stable':np.empty((0,2)), 'only':np.empty((0,2)), 'aomc':0}
            summary[county] = {'aomc_2021_pixels':0,'stable_pixels':0,'usable_after_300m':0}
            continue
        xx, yy = rasterio.transform.xy(tr, ys, xs, offset='center')
        xy = np.column_stack([np.asarray(xx), np.asarray(yy)])
        # Read 2019/2020 values at each 2021 apple-pixel center. Stable means >0 in all 3 years.
        v19 = np.asarray([v[0] for v in s19.sample(xy)], dtype=np.uint8) > 0
        v20 = np.asarray([v[0] for v in s20.sample(xy)], dtype=np.uint8) > 0
        stable = v19 & v20
        if excl_tree is not None:
            d,_ = excl_tree.query(xy, k=1)
            keep = d >= EXCL_DIST
        else:
            keep = np.ones(len(xy), bool)
        stable_xy = xy[stable & keep]
        only_xy = xy[(~stable) & keep]
        # Reproducible randomized order within priority groups.
        if len(stable_xy): stable_xy = stable_xy[rng.permutation(len(stable_xy))]
        if len(only_xy): only_xy = only_xy[rng.permutation(len(only_xy))]
        bycounty[county] = {'stable':stable_xy, 'only':only_xy, 'aomc':int(len(xy))}
        summary[county] = {'aomc_2021_pixels':int(len(xy)), 'stable_pixels':int(stable.sum()), 'usable_after_300m':int(keep.sum())}

# Quotas: proportional to 2021 apple pixels, minimum 5 for every county with apples,
# maximum 35 per county so no single county dominates. The zero-apple county is omitted.
active = [c for c in [f['properties']['name'] for f in counties] if bycounty[c]['aomc'] > 0]
quota = {c: 5 for c in active}
# Start with proportional rounded quotas, then enforce bounds and adjust to exact target.
raw = {c: TARGET*bycounty[c]['aomc']/sum(bycounty[x]['aomc'] for x in active) for c in active}
quota = {c:min(35,max(5,int(round(raw[c])))) for c in active}
while sum(quota.values()) < TARGET:
    choices=[c for c in active if quota[c]<35]
    c=max(choices, key=lambda z: raw[z]-quota[z])
    quota[c]+=1
while sum(quota.values()) > TARGET:
    choices=[c for c in active if quota[c]>5]
    c=min(choices, key=lambda z: raw[z]-quota[z])
    quota[c]-=1

# Greedy round-robin thinning ensures the county quotas and global 450m spacing.
selected=[]
selected_count={c:0 for c in active}
# Candidate iterators: stable first, then 2021-only. We keep a per-county pointer.
pools={c:np.vstack([bycounty[c]['stable'],bycounty[c]['only']]) if (len(bycounty[c]['stable'])+len(bycounty[c]['only'])) else np.empty((0,2)) for c in active}
prior_lens={c:len(bycounty[c]['stable']) for c in active}
ptr={c:0 for c in active}
# Test candidate against selected and exclusions. Exclusions already screened, but retained defensively.
def okay(p):
    if selected:
        sel=np.asarray(selected)
        if np.min(np.sqrt(((sel-p)**2).sum(axis=1))) < MIN_SEP: return False
    return True
# Iterate several passes; each pass gives every county one point toward its quota.
for round_i in range(max(quota.values())):
    for c in active:
        if selected_count[c] >= quota[c]: continue
        pool=pools[c]; found=False
        while ptr[c] < len(pool):
            p=pool[ptr[c]]; ptr[c]+=1
            if okay(p):
                selected.append(p); selected_count[c]+=1; found=True; break
        if not found:
            pass

# If thinning and clustering leave quota gaps, use any remaining pixel candidates, still respecting spacing.
if sum(selected_count.values()) < TARGET:
    all_remaining=[]
    for c in active:
        pool=pools[c]
        for p in pool[ptr[c]:]: all_remaining.append((c,p))
    rng.shuffle(all_remaining)
    for c,p in all_remaining:
        if sum(selected_count.values())>=TARGET: break
        if okay(p): selected.append(p); selected_count[c]=selected_count.get(c,0)+1

# Fail loudly rather than inventing points if source pixels cannot support target.
if sum(selected_count.values()) < 150:
    raise RuntimeError(f'Only {sum(selected_count.values())} points could be thinned; refusing to invent coordinates')
# Trim only if a pathological fallback exceeded target (not expected).
if len(selected)>TARGET:
    selected=selected[:TARGET]
# Attach metadata by county lookup / stable status.
# selected was round-robin; identify source priority from per-county arrays with coordinate tuple lookup.
meta=[]
for c in active:
    stset={tuple(p) for p in bycounty[c]['stable']}
    for p in selected:
        # assign county by exact coordinate membership in its county pool
        if any(np.array_equal(p,q) for q in pools[c]):
            pr='high' if tuple(p) in stset else 'normal'
            meta.append((c,p,pr)); break
# Above one-per-county is insufficient; do direct mapping with coordinate map.
coordmap={}
for c in active:
    for p in bycounty[c]['stable']: coordmap[tuple(p)]=(c,'high')
    for p in bycounty[c]['only']: coordmap[tuple(p)]=(c,'normal')
meta=[(coordmap[tuple(p)][0],p,coordmap[tuple(p)][1]) for p in selected]
# Preserve selection order and assign ids.
records=[]
for i,(c,p,pr) in enumerate(meta,1):
    lon,lat=Transformer.from_crs(s21.crs,'EPSG:4326',always_xy=True).transform(float(p[0]),float(p[1]))
    records.append({'id':f'YL-SURV-{i:04d}','lon':round(lon,7),'lat':round(lat,7),'county':c,'source':'AOMC_candidate','priority':pr,'note':NOTE})

# Write CSV and GeoJSON.
outstem=OUT/'Ili_apple_orchard_survey_candidates'
pd.DataFrame(records).to_csv(str(outstem)+'.csv',index=False,encoding='utf-8-sig')
features=[]
for r in records:
    props={k:r[k] for k in ['id','county','source','priority','note']}
    features.append({'type':'Feature','geometry':{'type':'Point','coordinates':[r['lon'],r['lat']]},'properties':props})
geo={'type':'FeatureCollection','name':'Ili_apple_orchard_survey_candidates','crs':{'type':'name','properties':{'name':'urn:ogc:def:crs:OGC::4326'}},'features':features}
(outstem.with_suffix('.geojson')).write_text(json.dumps(geo,ensure_ascii=False,indent=2),encoding='utf-8')
# KML, with county folders.
kml=[]
app=kml.append
app('<?xml version="1.0" encoding="UTF-8"?>')
app('<kml xmlns="http://www.opengis.net/kml/2.2"><Document>')
app('<name>Ili apple orchard survey candidates</name>')
for c in active:
    rs=[r for r in records if r['county']==c]
    if not rs: continue
    app(f'<Folder><name>{html.escape(c)}</name>')
    for r in rs:
        desc=f"县：{r['county']}<br/>优先级：{r['priority']}<br/>来源：{r['source']}<br/>{r['note']}"
        app('<Placemark>')
        app(f'<name>{r["id"]}</name><description><![CDATA[{desc}]]></description>')
        app(f'<Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>')
    app('</Folder>')
app('</Document></kml>')
(outstem.with_suffix('.kml')).write_text('\n'.join(kml),encoding='utf-8')

# README in Chinese, concise with counts and caveats.
counts=pd.Series([r['county'] for r in records]).value_counts().reindex(active,fill_value=0)
readme=['# 伊犁苹果园野外调查候选点','',f'共生成 **{len(records)}** 个候选点（目标175），点位均取自2021年AOMC栅格中值>0的真实像元中心；其中2019、2020、2021三期均为苹果像元的点标记为 `high`，其余2021候选点标记为 `normal`。','', '## 生成方法','- 将2021 AOMC苹果像元裁剪至伊犁州县域边界，并以2019/2020同位置像元值>0识别稳定点。','- 排除距已有可用苹果调查GPS点300 m以内的像元；候选点之间按450 m最小间距进行空间抽稀。','- 按县域AOMC苹果像元数近似比例分配，单县最低5点、最高35点；新疆源县、特克斯县、巩留县、伊宁县等有苹果像元的县均覆盖。','- 备注中的“候选”仅表示建议核查位置，不是地面真值。','', '## 各县数量','|县|数量|','|---|---:|']
for c,n in counts.items(): readme.append(f'|{c}|{int(n)}|')
readme += ['', '## 使用注意','AOMC可能存在错分（commission）和漏分（omission）；现场需核实是否为苹果园，并记录实际地类、果园状态与定位质量。昭苏县在2021 AOMC裁剪结果中无苹果像元，因此未分配点。','']
(OUT/'README_survey_candidates.md').write_text('\n'.join(readme),encoding='utf-8')

print('TOTAL',len(records))
print('QUOTA',quota)
print('SELECTED',selected_count)
print('COUNTS',json.dumps(counts.astype(int).to_dict(),ensure_ascii=False))
print('PRIORITY',pd.Series([r['priority'] for r in records]).value_counts().to_dict())
print('OUT',outstem.with_suffix('.kml'))
