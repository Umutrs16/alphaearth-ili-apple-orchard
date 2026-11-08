/**
 * 07_batch_years.js — 2019–2025 批量导出年度 NDVI_max（骨架）
 * GEE Project: hallowed-chain-458005-g9
 * 提示：GEE 对循环 Export 有配额限制，建议逐年 Run 或使用 Python API 批量提交任务。
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];

function maskS2(img) {
  var scl = img.select('SCL');
  return img.updateMask(scl.neq(8).and(scl.neq(9)).and(scl.neq(10))).divide(10000);
}

years.forEach(function(year) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
    .map(maskS2)
    .map(function(img) {
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        .copyProperties(img, ['system:time_start']);
    });
  var ndviMax = col.max().clip(aoi).rename('NDVI_max');
  Export.image.toDrive({
    image: ndviMax.toFloat(),
    description: 'NDVI_max_' + year,
    region: aoi,
    scale: 30,
    maxPixels: 1e13
  });
  print('Queued export stub for', year);
});

Map.centerObject(aoi, 9);
