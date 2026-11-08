/**
 * 06_annual_classify_export.js — 年度特征栈导出（供本地模型或 GEE 分类器）
 * GEE Project: hallowed-chain-458005-g9
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var year = 2021;

function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(mask).divide(10000);
}
function addIdx(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return img.addBands(ndvi);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2).map(addIdx);

var s2f = ee.Image.cat([
  s2.select('NDVI').max().rename('NDVI_max'),
  s2.select('NDVI').mean().rename('NDVI_mean'),
  s2.select('B8').mean().rename('B8_mean'),
  s2.select('B4').mean().rename('B4_mean')
]);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);
var s1f = ee.Image.cat([
  s1.select('VV').mean().rename('VV_mean'),
  s1.select('VH').mean().rename('VH_mean')
]);

var dem = ee.Image('USGS/SRTMGL1_003');
var demf = ee.Image.cat([
  dem.rename('elevation'),
  ee.Terrain.slope(dem).rename('slope')
]);

var ae = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(aoi)
  .filterDate(year + '-01-01', year + '-12-31')
  .mosaic();

var stack = s2f.addBands(s1f).addBands(demf).addBands(ae).clip(aoi).toFloat();
print('stack bands', stack.bandNames());

Map.centerObject(aoi, 9);
Map.addLayer(stack.select('NDVI_max'), {min: 0.2, max: 0.9}, 'stack_NDVI_max');

Export.image.toDrive({
  image: stack,
  description: 'FeatureStack_' + year,
  region: aoi,
  scale: 30,
  maxPixels: 1e13
});

// 若已在 GEE 训练 ee.Classifier，可在此 .classify(stack) 后导出年度园地概率/类别图：
// Export.image.toDrive({ image: classified, description: 'OrchardMap_' + year, ... });
