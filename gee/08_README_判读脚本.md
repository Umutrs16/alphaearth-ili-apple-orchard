# GEE 目视判读脚本

文件：`08_gongliu_visual_interpretation.js`

## 使用步骤
1. 打开 [Earth Engine Code Editor](https://code.earthengine.google.com/?project=hallowed-chain-458005-g9)
2. 新建脚本，粘贴本文件全部内容
3. 确认右上角 Cloud Project 为 `hallowed-chain-458005-g9`
4. 点击 Run
5. 左侧面板：上一点 / 下一点 / 按编号跳转 / 按分层筛选
6. 左右地图联动；可改左右年份后点「重载」
7. 对照当前 `sample_id`，在桌面 Excel 判读表填写 code

## 说明
- 脚本**不会**自动写类别，避免循环验证
- 点数据已内嵌，无需先上传 Asset
- 若以后上传了 `Gongliu_IV_points` 资产，可将 `USE_EMBEDDED` 改为 `false`
