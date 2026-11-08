/**
 * 02_s2_phenology.js — Sentinel-2 云掩膜、指数与年物候统计
 * GEE Project: hallowed-chain-458005-g9
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var year = 2021; // 修改年份

function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  var qa = img.select('QA60');
  var cloudBit = 1 << 10;
  var cirrusBit = 1 << 11;
  var maskQa = qa.bitwiseAnd(cloudBit).eq(0).and(qa.bitwiseAnd(cirrusBit).eq(0));
  return img.updateMask(mask.and(maskQa))
    .divide(10000)
    .copyProperties(img, ['system:time_start']);
}

function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
    {NIR: img.select('B8'), RED: img.select('B4'), BLUE: img.select('B2')}
  ).rename('EVI');
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndre = img.normalizedDifference(['B8', 'B5']).rename('NDRE');
  return img.addBands([ndvi, evi, ndwi, ndre]);
}

var start = ee.Date.fromYMD(year, 1, 1);
var end = ee.Date.fromYMD(year, 12, 31);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(start, end)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2)
  .map(addIndices);

var ndviCol = s2.select('NDVI');
var phenology = ee.Image.cat([
  ndviCol.max().rename('NDVI_max'),
  ndviCol.mean().rename('NDVI_mean'),
  ndviCol.reduce(ee.Reducer.stdDev()).rename('NDVI_std'),
  ndviCol.reduce(ee.Reducer.percentile([10, 90])),
  s2.select('EVI').max().rename('EVI_max'),
  s2.select('EVI').mean().rename('EVI_mean'),
  s2.select('NDWI').mean().rename('NDWI_mean'),
  s2.select('NDRE').mean().rename('NDRE_mean'),
  s2.select('B4').mean().rename('B4_mean'),
  s2.select('B8').mean().rename('B8_mean'),
  s2.select('B11').mean().rename('B11_mean'),
  s2.select('B12').mean().rename('B12_mean')
]).clip(aoi);

// 振幅
phenology = phenology.addBands(
  phenology.select('NDVI_max').subtract(phenology.select('NDVI_p10')).rename('NDVI_amp')
);

Map.centerObject(aoi, 9);
Map.addLayer(phenology.select('NDVI_max'), {min: 0.2, max: 0.9, palette: ['white', 'green']}, 'NDVI_max_' + year);

Export.image.toDrive({
  image: phenology.toFloat(),
  description: 'S2_phenology_' + year,
  region: aoi,
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
