/**
 * 05_aomc_sample_export.js — 叠加特征并导出带空间块 ID 的训练样本表
 * GEE Project: hallowed-chain-458005-g9
 *
 * 使用前：将纯化后的 AOMC 园地栅格上传为 Asset，或用已有园地概率/掩膜替换 orchardMask。
 * 本脚本提供可运行的采样骨架（以 NDVI_max 阈值生成示意掩膜，正式请替换为 AOMC）。
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var year = 2021;
var blockSize = 0.05; // 度
var nPos = 2000;
var nNeg = 2000;
var scale = 30;

function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(mask).divide(10000);
}
function addNDVI(img) {
  return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'));
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2).map(addNDVI);
var ndviMax = s2.select('NDVI').max().clip(aoi).rename('NDVI_max');

// TODO: 替换为 AOMC Asset，例如：
// var aomc = ee.Image('users/YOUR_USER/AOMC_30_Xinjiang_2021').clip(aoi);
// var orchardMask = aomc.gt(0);
var orchardMask = ndviMax.gt(0.65); // 示意占位，非正式标签

var dem = ee.Image('USGS/SRTMGL1_003').clip(aoi);
var elev = dem.rename('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');

var stack = ndviMax.addBands([elev, slope]);

// 空间块 ID：floor(lon/bs)*1e5 + floor(lat/bs)
var lon = ee.Image.pixelLonLat().select('longitude');
var lat = ee.Image.pixelLonLat().select('latitude');
var blockId = lon.divide(blockSize).floor().multiply(1e5)
  .add(lat.divide(blockSize).floor()).toInt().rename('block_id');
stack = stack.addBands(blockId);

var pos = stack.updateMask(orchardMask).sample({
  region: aoi, scale: scale, numPixels: nPos, seed: 42, geometries: true
}).map(function(f) { return f.set('label', 1).set('year', year); });

var neg = stack.updateMask(orchardMask.not()).sample({
  region: aoi, scale: scale, numPixels: nNeg, seed: 43, geometries: true
}).map(function(f) { return f.set('label', 0).set('year', year); });

var samples = pos.merge(neg);
print('sample size', samples.size());
Map.centerObject(aoi, 9);
Map.addLayer(orchardMask.selfMask(), {palette: ['green']}, 'orchardMask_proxy');

Export.table.toDrive({
  collection: samples,
  description: 'AOMC_like_samples_' + year,
  fileFormat: 'CSV'
});
