/**
 * 03_s1_features.js — Sentinel-1 IW VV+VH 年统计与 RVI
 * GEE Project: hallowed-chain-458005-g9
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var year = 2021;

function toDB(img) {
  // S1_GRD 已是 dB；保证波段存在
  return img.select(['VV', 'VH']).copyProperties(img, ['system:time_start']);
}

function addRVI(img) {
  // 线性域近似：先从 dB 转线性
  var vv = ee.Image(10).pow(img.select('VV').divide(10)).rename('VVlin');
  var vh = ee.Image(10).pow(img.select('VH').divide(10)).rename('VHlin');
  var rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI');
  return img.addBands(rvi);
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .map(toDB)
  .map(addRVI);

var s1feat = ee.Image.cat([
  s1.select('VV').mean().rename('VV_mean'),
  s1.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_std'),
  s1.select('VH').mean().rename('VH_mean'),
  s1.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_std'),
  s1.select('RVI').mean().rename('RVI')
]).clip(aoi);

s1feat = s1feat.addBands(
  s1feat.select('VV_mean').subtract(s1feat.select('VH_mean')).rename('VV_VH_ratio')
);

Map.centerObject(aoi, 9);
Map.addLayer(s1feat.select('VV_mean'), {min: -18, max: -5}, 'VV_mean_' + year);

Export.image.toDrive({
  image: s1feat.toFloat(),
  description: 'S1_features_' + year,
  region: aoi,
  scale: 30,
  maxPixels: 1e13
});
