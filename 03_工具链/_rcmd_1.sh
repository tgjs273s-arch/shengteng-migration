#!/bin/bash
# 远端调查①：定位 mindspeed_mm 的数据通路（**只读**，不改远端任何文件）
P=$(python3 -c 'import mindspeed_mm,os;print(os.path.dirname(mindspeed_mm.__file__))')
echo "PKG=$P"
echo "--- top level ---"
ls "$P" | head -40
echo "--- 数据相关 .py（maxdepth 3）---"
find "$P" -maxdepth 3 -name '*.py' | grep -iE 'data|dataset|loader|collate|sampler' | head -40
