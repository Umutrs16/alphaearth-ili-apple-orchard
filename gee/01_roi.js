/**
 * 01_roi.js — 巩留核心 AOI / 可扩展伊犁河谷
 * GEE Project: hallowed-chain-458005-g9
 * 使用：粘贴到 Earth Engine Code Editor，确认项目后 Run
 */
// ee.Initialize({project: 'hallowed-chain-458005-g9'}); // Python 端；Code Editor 在右上角选项目

var geeProject = 'hallowed-chain-458005-g9';

var gongliuCore = ee.Geometry.Rectangle([82.0, 43.1, 82.8, 43.6], 'EPSG:4326', false);
var iliValley = ee.Geometry.Rectangle([80.5, 42.8, 85.0, 44.5], 'EPSG:4326', false);

var aoi = gongliuCore; // 切换为 iliValley 以扩展

Map.centerObject(aoi, 9);
Map.addLayer(aoi, {color: 'FF0000'}, 'Gongliu_core_AOI');
Map.addLayer(iliValley, {color: '0000FF'}, 'Ili_Valley_expandable', false);

print('GEE project (set in Code Editor):', geeProject);
print('AOI area m2', aoi.area(1));

// 导出 AOI 为 Asset / GeoJSON 表
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(aoi, {name: 'Gongliu_core'})]),
  description: 'Yili_Gongliu_AOI',
  fileFormat: 'GeoJSON'
});
