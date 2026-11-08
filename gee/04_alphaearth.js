/**
 * 04_alphaearth.js — AlphaEarth / Satellite Embedding V1 Annual 镶嵌
 * GEE Project: hallowed-chain-458005-g9
 * 数据集: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
 * 注意：波段数以 Data Catalog 最新元数据为准；本脚本做通用镶嵌与裁剪。
 */
var geeProject = 'hallowed-chain-458005-g9';
var aoi = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var year = 2021;

var aeCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(aoi)
  .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31));

// 若按 calendar year 属性过滤更稳妥，可追加：
// .filter(ee.Filter.eq('year', year))  // 视目录属性名调整

var ae = aeCol.mosaic().clip(aoi);
print('AlphaEarth band names', ae.bandNames());
print('AlphaEarth image count', aeCol.size());

Map.centerObject(aoi, 9);
// 仅可视化前 3 个嵌入维
Map.addLayer(ae, {bands: ae.bandNames().slice(0, 3), min: -0.5, max: 0.5}, 'AE_RGB_proxy_' + year);

Export.image.toDrive({
  image: ae.toFloat(),
  description: 'AlphaEarth_annual_' + year,
  region: aoi,
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
